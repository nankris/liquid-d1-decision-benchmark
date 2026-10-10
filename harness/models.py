"""Endpoint registry and clients. No network activity occurs during import."""
from dataclasses import dataclass
import math
import os
import time
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from common import DATA, read_jsonl

@dataclass(frozen=True)
class Endpoint:
    kind: str
    base: str
    env: str
    key: str = ''
    model: str = ''

# A new fixed endpoint needs only one entry here.
REGISTRY = {
    'd1-local': Endpoint('decision', 'http://127.0.0.1:8081', 'D1_LOCAL_BASE_URL'),
    'qwen-local': Endpoint('chat', 'http://127.0.0.1:8082', 'QWEN_LOCAL_BASE_URL', model='Qwen2.5-3B-Instruct'),
    'd1-hosted': Endpoint('decision', 'https://api.liquid.ai/decisions', 'LIQUID_BASE_URL', 'LIQUID_API_KEY', 'd1'),
    'openrouter-d1': Endpoint('decision', 'https://openrouter.ai/api', 'OPENROUTER_BASE_URL', 'OPENROUTER_API_KEY', 'liquid/d1'),
}

class SkipModel(Exception):
    pass

class CallError(Exception):
    def __init__(self, message, response, latency_ms, transient=False, unsupported=False):
        super().__init__(message)
        self.response, self.latency_ms = response, latency_ms
        self.transient, self.unsupported = transient, unsupported


def distribution(values, options):
    probs = {option: float(values.get(option, 0)) for option in options}
    if any(not math.isfinite(p) or p < 0 for p in probs.values()):
        raise ValueError('Invalid probability values')
    total = sum(probs.values())
    return ({k: v / total for k, v in probs.items()}, False) if total else (dict.fromkeys(options, 1 / len(options)), True)


def parse_chat(response, task, allow_fallback=False):
    choice = response['choices'][0]
    content = choice.get('message', {}).get('content') or ''
    parsed = content.strip().lower()
    mapping = {token.lower(): label for token, label in task.tokens.items()}
    text_valid = parsed in mapping
    entries = ((choice.get('logprobs') or {}).get('content') or [])
    tops = entries[0].get('top_logprobs') if entries else None
    fallback = not tops
    if fallback and not allow_fallback:
        raise ValueError('Local chat response did not contain first-token top_logprobs')
    masses = dict.fromkeys(task.options, 0.0)
    if fallback:
        if text_valid:
            masses[mapping[parsed]] = 1.0
    else:
        for item in tops:
            token = item['token'].strip().lower()
            if token in mapping:
                masses[mapping[token]] += math.exp(float(item['logprob']))
    probs, invalid = distribution(masses, task.options)
    pred = max(probs, key=probs.get)
    return {'pred': pred, 'probs': probs, 'valid': text_valid and not invalid,
            'invalid': invalid or not text_valid, 'parsed_text': parsed, 'text_valid': text_valid,
            'parsed_answer': mapping.get(parsed), 'text_only_fallback': fallback,
            'expected_score': sum(int(k) * p for k, p in probs.items()) if task.kind == 'score' else None}


def parse_decision(response, task, name='answer'):
    answers = response.get('answers', response.get('results', response))
    answer = answers[name]
    if task.kind == 'noul':
        p = float(answer['noul'])
        if not 0 <= p <= 1:
            raise ValueError('noul probability outside [0,1]')
        values = {'0': 1 - p, '1': p}
    else:
        values = answer['probabilities']
    probs, invalid = distribution(values, task.options)
    expected = float(answer['score']) if task.kind == 'score' else None
    if expected is not None and (not math.isfinite(expected) or not 0 <= expected <= len(task.options) - 1):
        raise ValueError('Score outside criteria range')
    return {'pred': max(probs, key=probs.get), 'probs': probs, 'valid': not invalid,
            'invalid': invalid, 'expected_score': expected, 'text_only_fallback': False}


class HTTPClient:
    def __init__(self, name):
        self.name = name
        self.endpoint = (Endpoint('chat', 'https://openrouter.ai/api', 'OPENROUTER_BASE_URL',
                                  'OPENROUTER_API_KEY', name.split(':', 1)[1])
                         if name.startswith('openrouter:') else REGISTRY[name])
        e = self.endpoint
        self.paid = bool(e.key)
        key = os.getenv(e.key) if e.key else None
        if e.key and not key:
            raise SkipModel(f'Skipping {name}: {e.key} is missing.')
        base = os.getenv(e.env, e.base).rstrip('/')
        self.url = base + ('/systemone' if base.endswith('/v1') else '/v1/systemone') if e.kind == 'decision' else base + ('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')
        self.session = requests.Session()
        if key:
            self.session.headers['Authorization'] = f'Bearer {key}'
        self.text_only = False

    def body(self, task, text):
        e = self.endpoint
        if e.kind == 'decision':
            body = {'state': text, 'questions': {'answer': task.question()}}
            if e.model:
                body['model'] = os.getenv('LIQUID_MODEL', e.model) if self.name == 'd1-hosted' else e.model
        else:
            body = {'model': e.model, 'messages': task.messages(text), 'temperature': 0, 'max_tokens': 4}
            if not self.text_only:
                body.update(logprobs=True, top_logprobs=20)
        return body

    def call(self, body):
        start = time.perf_counter()
        try:
            response = self.session.post(self.url, json=body, timeout=(10, 300))
        except requests.RequestException as exc:
            raise CallError(str(exc), None, (time.perf_counter() - start) * 1000, transient=True) from exc
        elapsed = (time.perf_counter() - start) * 1000
        try:
            payload = response.json()
        except ValueError:
            payload = {'text': response.text}
        if not response.ok:
            message = f'HTTP {response.status_code}: {payload}'
            unsupported = self.name.startswith('openrouter:') and response.status_code in (400, 404, 422) and 'logprob' in str(payload).lower()
            raise CallError(message, payload, elapsed, response.status_code in (408, 425, 429) or response.status_code >= 500, unsupported)
        return payload, elapsed

    def parse(self, response, task):
        parsed = parse_decision(response, task) if self.endpoint.kind == 'decision' else parse_chat(response, task, self.paid)
        usage = response.get('usage') or {}
        parsed.update(input_tokens=usage.get('input_tokens', usage.get('prompt_tokens', 0)),
                      output_tokens=usage.get('output_tokens', usage.get('completion_tokens', 0)),
                      actual_cost_usd=usage.get('cost'))
        return parsed


class TfidfClient:
    paid = False
    def __init__(self, task):
        train = read_jsonl(DATA / f'{task.name}_train.jsonl')
        if not train:
            raise ValueError(f'Missing training data for {task.name}; run prepare_data.py')
        self.pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                      LogisticRegression(max_iter=2000, random_state=42))
        self.pipeline.fit([r['text'] for r in train], [str(r['label']) for r in train])

    def body(self, task, text):
        return {'text': text, 'task': task.name}

    def call(self, body):
        start = time.perf_counter()
        values = self.pipeline.predict_proba([body['text']])[0]
        elapsed = (time.perf_counter() - start) * 1000
        return {'probabilities': dict(zip(self.pipeline.classes_, values.tolist()))}, elapsed

    def parse(self, response, task):
        probs, invalid = distribution(response['probabilities'], task.options)
        return {'pred': max(probs, key=probs.get), 'probs': probs, 'valid': not invalid, 'invalid': invalid,
                'expected_score': sum(int(k) * p for k, p in probs.items()) if task.kind == 'score' else None,
                'input_tokens': 0, 'output_tokens': 0, 'text_only_fallback': False}


def make_client(model, task):
    return TfidfClient(task) if model == 'tfidf-lr' else HTTPClient(model)
