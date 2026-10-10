"""Summarize successful unique evaluations; report failed attempts separately."""
import json
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from common import ROOT, RAW, read_jsonl, successful
from prices import cost
from tasks import TASKS

RESULTS = ROOT / 'results'
BASE_COLUMNS = ['model', 'task', 'n', 'n_errors', 'n_invalid', 'accuracy', 'accuracy_ci_low', 'accuracy_ci_high',
                'macro_f1', 'ece', 'latency_p50_ms', 'latency_p95_ms', 'latency_mean_ms', 'mean_input_tokens', 'mean_output_tokens']
ITEM_COLUMNS = ['model', 'task', 'id', 'label', 'pred', 'correct', 'valid', 'latency_ms', 'input_tokens',
                'output_tokens', 'confidence', 'p_positive', 'expected_score', 'probs', 'text_only_fallback']

def calibration(probabilities, outcomes):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    indices = np.minimum((p * 10).astype(int), 9)
    points = []
    for b in range(10):
        mask = indices == b
        if mask.any():
            points.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    ece = sum(abs(p - y) * n for p, y, n in points) / len(p) if len(p) else None
    return ece, points


def groups():
    grouped = defaultdict(list)
    for path in sorted(RAW.glob('*.jsonl')):
        for row in read_jsonl(path):
            if row.get('task') in TASKS:
                grouped[row['model'], row['task']].append(row)
    return grouped


def summarize(model, task_name, attempts):
    task = TASKS[task_name]
    rows = successful(attempts)
    n = len(rows)
    result = dict.fromkeys(BASE_COLUMNS)
    result.update(model=model, task=task_name, n=n,
                  n_errors=sum(bool(r.get('error')) for r in attempts if not r.get('warmup')),
                  n_invalid=sum(not r.get('valid', False) for r in rows),
                  n_text_only_fallback=sum(bool(r.get('text_only_fallback')) for r in rows))
    if not n:
        return result, []
    y = np.array([str(r['label']) for r in rows])
    pred = np.array([str(r['pred']) for r in rows])
    correct = y == pred
    probabilities = np.array([[r['probs'].get(option, 0) for option in task.options] for r in rows])
    confidence = probabilities.max(axis=1)
    rng = np.random.default_rng(0)
    boot = correct[rng.integers(0, n, size=(1000, n))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    result.update(accuracy=float(correct.mean()), accuracy_ci_low=float(lo), accuracy_ci_high=float(hi),
                  macro_f1=float(f1_score(y, pred, labels=task.options, average='macro', zero_division=0)),
                  ece=calibration(confidence, correct)[0])
    expected = np.array([r.get('expected_score') if r.get('expected_score') is not None else
                         sum(int(k) * p for k, p in r['probs'].items()) if task.kind == 'score' else np.nan for r in rows])
    if task.kind == 'noul':
        positive = (y == '1').astype(int)
        p = probabilities[:, task.options.index('1')]
        precision, recall, f1, _ = precision_recall_fscore_support(positive, (pred == '1').astype(int), average='binary', zero_division=0)
        result.update(precision=float(precision), recall=float(recall), f1=float(f1),
                      roc_auc=float(roc_auc_score(positive, p)) if len(set(positive)) > 1 else None,
                      brier=float(np.mean((p - positive) ** 2)), ece=calibration(p, positive)[0])
    if task.kind == 'score':
        true = y.astype(int)
        argmax = pred.astype(int)
        rounded = np.clip(np.floor(expected + 0.5), 0, 4).astype(int)
        result.update(exact_accuracy=float(correct.mean()), within_1_accuracy=float(np.mean(abs(true - argmax) <= 1)),
                      mae_stars=float(np.mean(abs(true - argmax))),
                      expected_rounded_accuracy=float(np.mean(true == rounded)),
                      expected_rounded_within_1_accuracy=float(np.mean(abs(true - rounded) <= 1)),
                      expected_rounded_mae_stars=float(np.mean(abs(true - rounded))),
                      expected_score_mae_stars=float(np.mean(abs(true - expected))),
                      spearman=float(pd.Series(true).corr(pd.Series(expected), method='spearman')) if len(set(true)) > 1 and len(set(expected)) > 1 else None)
    latencies = [r['latency_ms'] for r in rows if r.get('latency_ms') is not None]
    if latencies:
        result.update(latency_p50_ms=float(np.percentile(latencies, 50)), latency_p95_ms=float(np.percentile(latencies, 95)), latency_mean_ms=float(np.mean(latencies)))
    input_mean = float(np.mean([r.get('input_tokens', 0) or 0 for r in rows]))
    output_mean = float(np.mean([r.get('output_tokens', 0) or 0 for r in rows]))
    result.update(mean_input_tokens=input_mean, mean_output_tokens=output_mean)
    projected = cost(model, input_mean * 1000, output_mean * 1000)
    result['projected_usd_per_1000'] = projected
    result['cost_projection_note'] = ('Projection: d1 $0.04/M input; local and hosted tokenizer family assumed comparable.'
                                      if model in ('d1-local', 'd1-hosted', 'openrouter-d1') else
                                      'Projection using configured input/output token prices.' if projected is not None else None)
    items = []
    for i, row in enumerate(rows):
        item = {key: row.get(key) for key in ITEM_COLUMNS}
        item.update(label=str(row['label']), pred=str(row['pred']), correct=bool(correct[i]), confidence=float(confidence[i]),
                    p_positive=float(probabilities[i, 1]) if task.kind == 'noul' else None,
                    expected_score=float(expected[i]) if task.kind == 'score' else None,
                    probs=json.dumps(row['probs'], sort_keys=True))
        items.append(item)
    return result, items


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries, items = [], []
    for (model, task), attempts in sorted(groups().items()):
        summary, detail = summarize(model, task, attempts)
        summaries.append(clean(summary))
        items.extend(detail)
    frame = pd.DataFrame(summaries) if summaries else pd.DataFrame(columns=BASE_COLUMNS)
    frame.to_csv(RESULTS / 'summary.csv', index=False)
    (RESULTS / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    pd.DataFrame(items, columns=ITEM_COLUMNS).to_csv(RESULTS / 'per_item.csv', index=False)
    if summaries:
        columns = ['model', 'task', 'n', 'n_errors', 'n_invalid', 'accuracy', 'macro_f1', 'ece', 'latency_p50_ms', 'latency_p95_ms']
        table = frame[columns].to_string(index=False, float_format=lambda x: f'{x:.3f}')
        table += '\n\nAccuracy CIs: 1,000 bootstrap resamples, seed 0. Warmups excluded.\n'
        projections = frame[frame.get('projected_usd_per_1000', pd.Series(dtype=float)).notna()] if 'projected_usd_per_1000' in frame else pd.DataFrame()
        if not projections.empty:
            table += '\nHosted cost projection (USD/1,000 decisions):\n' + projections[['model', 'task', 'projected_usd_per_1000']].to_string(index=False) + '\nD1 assumes comparable local/hosted tokenizer families; $0.04/M input tokens.\n'
    else:
        table = 'No evaluation results found. Run harness/run.py first.\n'
    print(table)
    (RESULTS / 'summary.txt').write_text(table)

if __name__ == '__main__':
    main()
