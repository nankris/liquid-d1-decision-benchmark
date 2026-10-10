"""Offline regression tests; no real HTTP calls or paid endpoints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analyze import calibration, summarize
from common import read_jsonl, successful
from models import CallError, HTTPClient, parse_chat, parse_decision
from prepare_data import replace_placeholders, truncate
from run import Budget, BudgetExceeded, execute
from tasks import TASKS


def chat(text, pairs=None):
    return {'choices': [{'message': {'content': text}, 'logprobs': None if pairs is None else
                         {'content': [{'top_logprobs': [{'token': token, 'logprob': float(np.log(p))} for token, p in pairs]}]}}]}

class ParsingTests(unittest.TestCase):
    def test_first_token_mass_merging(self):
        response = chat('Yes', [('yes', .2), (' Yes', .3), ('NO', .1), ('maybe', .4)])
        response['choices'][0]['logprobs']['content'].append({'top_logprobs': [{'token': 'no', 'logprob': 0}]})
        result = parse_chat(response, TASKS['spam'])
        self.assertAlmostEqual(result['probs']['1'], 5 / 6)
        self.assertTrue(result['valid'])

    def test_missing_option_and_uniform(self):
        r = parse_chat(chat('yes', [(' Yes', .3)]), TASKS['spam'])
        self.assertEqual(r['probs'], {'0': 0, '1': 1})
        r = parse_chat(chat('maybe', [('unknown', 1)]), TASKS['spam'])
        self.assertEqual(r['probs'], {'0': .5, '1': .5})
        self.assertTrue(r['invalid'])

    def test_fallback_and_invalid_text(self):
        self.assertRaises(ValueError, parse_chat, chat('yes'), TASKS['spam'])
        r = parse_chat(chat('B'), TASKS['routing'], True)
        self.assertTrue(r['text_only_fallback'])
        self.assertEqual(r['pred'], 'CANCEL')
        r = parse_chat(chat('yes indeed', [('yes', .8), ('no', .2)]), TASKS['spam'])
        self.assertFalse(r['valid'])
        r = parse_chat(chat('???'), TASKS['rating'], True)
        self.assertTrue(r['invalid'])
        self.assertAlmostEqual(sum(r['probs'].values()), 1)

    def test_score_argmax_and_expected(self):
        r = parse_decision({'answers': {'answer': {'score': 1.6, 'probabilities': {'0': .2, '1': .1, '2': .6, '3': .1}}}}, TASKS['rating'])
        self.assertEqual(r['pred'], '2')
        self.assertEqual(r['expected_score'], 1.6)

    def test_preparation_helpers(self):
        self.assertEqual(replace_placeholders('Order {{Order Number}}, {{Unknown}}'), 'Order #48213, Unknown')
        self.assertLessEqual(len(truncate('word ' * 300)), 1000)
        self.assertTrue(truncate('word ' * 300).endswith('…'))

class MetricsTests(unittest.TestCase):
    def test_ece_boundaries(self):
        value, bins = calibration([0, .1, 1], [0, 1, 1])
        self.assertAlmostEqual(value, .3)
        self.assertEqual(sum(b[2] for b in bins), 3)

    def test_warmup_retry_dedup_and_rating(self):
        row = {'id': 'one', 'model': 'd1-local', 'task': 'rating', 'label': 1, 'pred': '0',
               'probs': {'0': .4, '1': .35, '2': .25, '3': 0, '4': 0}, 'expected_score': .85,
               'valid': True, 'error': None, 'latency_ms': 10, 'input_tokens': 100, 'output_tokens': 0}
        attempts = [dict(row, warmup=True), dict(row, error='timeout'), row, row]
        s, items = summarize('d1-local', 'rating', attempts)
        self.assertEqual((s['n'], s['n_errors'], len(items)), (1, 1, 1))
        self.assertEqual(s['accuracy'], 0)
        self.assertEqual(s['expected_rounded_accuracy'], 1)
        self.assertIsNone(s['spearman'])
        self.assertAlmostEqual(s['projected_usd_per_1000'], .004)

class RunnerTests(unittest.TestCase):
    def test_retry_every_attempt_logged(self):
        class Fake:
            paid = False
            n = 0
            def body(self, task, text):
                return {'state': text}
            def call(self, body):
                self.n += 1
                if self.n == 1:
                    raise CallError('HTTP 503', {'error': 'busy'}, 12, transient=True)
                return {'answers': {'answer': {'noul': .8}}, 'usage': {'input_tokens': 5}}, 8
            def parse(self, response, task):
                return parse_decision(response, task)
        with tempfile.TemporaryDirectory() as tmp, patch('run.time.sleep'):
            path = Path(tmp) / 'calls.jsonl'
            budget = Budget('d1-local', False, 2)
            execute(Fake(), TASKS['spam'], {'id': 'x', 'label': 1, 'text': 'hello'}, path, budget)
            rows = read_jsonl(path)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]['error'])
            self.assertEqual(rows[1]['input_tokens'], 5)
            self.assertEqual(len(successful(rows)), 1)
            self.assertRaises(BudgetExceeded, budget.before, {})

    def test_paid_guard_before_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch('run.RAW', Path(tmp)), patch.dict('os.environ', {'MAX_USD': '0.000001'}):
            budget = Budget('d1-hosted', True, 3)
            self.assertRaises(BudgetExceeded, budget.before, {'state': 'hello'})
            self.assertEqual(budget.calls, 0)
            self.assertRaises(BudgetExceeded, Budget, 'openrouter:unknown/model', True, 3)

    def test_existing_paid_spend(self):
        with tempfile.TemporaryDirectory() as tmp, patch('run.RAW', Path(tmp)), patch.dict('os.environ', {'MAX_USD': '2'}):
            (Path(tmp) / 'prior.jsonl').write_text(json.dumps({'model': 'd1-hosted', 'charged_cost_usd': 2}) + '\n')
            budget = Budget('d1-hosted', True, 3)
            self.assertRaises(BudgetExceeded, budget.before, {})

    def test_endpoints_construct_without_requests(self):
        with patch.dict('os.environ', {'LIQUID_API_KEY': 'test-key', 'OPENROUTER_API_KEY': 'test-key'}):
            client = HTTPClient('d1-hosted')
            self.assertEqual(client.url, 'https://api.liquid.ai/decisions/v1/systemone')
            self.assertEqual(client.body(TASKS['spam'], 'text')['model'], 'd1')
            self.assertEqual(HTTPClient('openrouter-d1').url, 'https://openrouter.ai/api/v1/systemone')
            self.assertEqual(HTTPClient('openrouter:provider/model').url, 'https://openrouter.ai/api/v1/chat/completions')

if __name__ == '__main__':
    unittest.main()
