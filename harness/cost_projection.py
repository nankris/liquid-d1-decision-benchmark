"""Projected API cost per 1,000 decisions, from LOGGED token counts x list prices.
d1: logged d1-3B input tokens x $0.04/M (Liquid blog + OpenRouter, Oct 2026), output free.
Chat models: logged Qwen2.5-3B prompt+completion tokens for the identical prompt, x OpenRouter list prices
(fetched from https://openrouter.ai/api/v1/models on 2026-10-10 10:59 UTC). Different tokenizers => approximate.
Chat projection is a LOWER bound: it assumes no reasoning tokens and a 1-2 token answer."""
import json, glob, csv, collections
PRICES = {  # USD per 1M tokens (input, output), OpenRouter list, 2026-10-10
    'd1 (hosted, list)': (0.04, 0.0),
    'gpt-6-luna': (0.10, 0.50),
    'claude-haiku-5.5': (0.10, 0.50),
    'gemini-3.1-flash-lite': (0.25, 1.50),
    'gpt-6.1-sol': (2.00, 10.00),
    'claude-opus-5.5': (4.00, 20.00),
}
def mean_tokens(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if not r.get('warmup') and not r.get('error')]
    return sum(r['input_tokens'] for r in rows) / len(rows), sum(r['output_tokens'] for r in rows) / len(rows)
out = []
for task in ['spam', 'toxic', 'routing', 'rating']:
    d_in, _ = mean_tokens(f'results/raw/d1-local__{task}.jsonl')
    q_in, q_out = mean_tokens(f'results/raw/qwen-local__{task}.jsonl')
    for name, (pi, po) in PRICES.items():
        tin, tout = (d_in, 0.0) if name.startswith('d1') else (q_in, q_out)
        usd = (tin * pi + tout * po) / 1e6 * 1000
        out.append({'task': task, 'model': name, 'mean_in_tokens': round(tin, 1), 'mean_out_tokens': round(tout, 1), 'usd_per_1000': usd})
base = {r['task']: r['usd_per_1000'] for r in out if r['model'].startswith('d1')}
for r in out: r['x_vs_d1'] = round(r['usd_per_1000'] / base[r['task']], 1)
with open('results/cost_projection.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out: print(f"{r['task']:8s} {r['model']:22s} in={r['mean_in_tokens']:6.1f} out={r['mean_out_tokens']:3.1f}  ${r['usd_per_1000']:.4f}/1k  {r['x_vs_d1']}x")
