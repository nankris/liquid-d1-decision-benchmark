"""Fairness check: Qwen2.5-3B on routing when asked to answer with the label NAME instead of a letter.
Text-only parse (no probabilities). Logs every call to results/raw/qwen-local-names__routing.jsonl."""
import json, time, datetime, requests, sys
sys.path.insert(0, 'harness')
from tasks import TASKS, CATEGORIES, SYSTEM
task = TASKS['routing']
items = [json.loads(l) for l in open('data/routing.jsonl')]
out = open('results/raw/qwen-local-names__routing.jsonl', 'a')
opts = '\n'.join(f'{k} — {v}' for k, v in CATEGORIES.items())
correct = 0
for i, it in enumerate(items):
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f"{task.instructions}\nReply with exactly one of these team names:\n{opts}\n\nText:\n{it['text']}\n\nAnswer:"}]
    body = {'messages': msgs, 'temperature': 0, 'max_tokens': 8}
    t = time.perf_counter(); r = requests.post('http://127.0.0.1:8082/v1/chat/completions', json=body, timeout=300).json(); ms = (time.perf_counter() - t) * 1000
    txt = r['choices'][0]['message']['content'].strip().upper().strip('.').split()[0] if r['choices'][0]['message']['content'].strip() else ''
    pred = txt if txt in CATEGORIES else None
    correct += pred == it['label']
    out.write(json.dumps({'ts_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model': 'qwen-local-names', 'task': 'routing', 'id': it['id'], 'label': it['label'], 'request': body, 'response': r, 'latency_ms': ms, 'pred': pred, 'valid': pred is not None}) + '\n')
print(f'qwen-local (label names) routing accuracy: {correct}/{len(items)} = {correct/len(items):.3f}')
