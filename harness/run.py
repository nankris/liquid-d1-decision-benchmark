"""Sequential, append-only and resumable evaluation runner."""
import argparse
import json
import math
import os
import time
from common import DATA, RAW, append, log_path, now, read_jsonl, successful
from models import CallError, SkipModel, make_client
from prices import cost, rates
from tasks import TASKS

class BudgetExceeded(Exception):
    pass

class Budget:
    def __init__(self, model, paid, max_calls):
        self.model, self.paid, self.max_calls = model, paid, max_calls
        self.calls = 0
        self.spent = 0.0
        self.ceiling = float(os.getenv('MAX_USD', '2'))
        if max_calls < 1 or not math.isfinite(self.ceiling) or self.ceiling <= 0:
            raise ValueError('max-calls and MAX_USD must be positive and finite')
        if paid and rates(model) is None:
            raise BudgetExceeded(f'No price configured for {model}; add input/output prices to harness/prices.py before paid calls.')
        if paid and any(not math.isfinite(rates(model)[key]) or rates(model)[key] < 0 for key in ('input', 'output')):
            raise BudgetExceeded('Configured input/output prices must be finite and nonnegative.')
        # Include previous runs for this model, including warmups and failed attempts.
        if paid:
            for path in RAW.glob('*.jsonl'):
                for row in read_jsonl(path):
                    if row.get('model') == model:
                        self.spent += float(row.get('charged_cost_usd') or 0)

    def before(self, body):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f'--max-calls={self.max_calls} reached (includes warmups and retries).')
        reserve = 0.0
        if self.paid:
            # UTF-8 byte count plus a generous framing allowance is a conservative
            # token estimate; reserve it before sending, even for failed calls.
            reserve = cost(self.model, len(json.dumps(body, ensure_ascii=False).encode()) + 4096, body.get('max_tokens', 0))
            if self.spent + reserve > self.ceiling:
                raise BudgetExceeded(f'Estimated spend would exceed MAX_USD=${self.ceiling:.2f}; stopping before request.')
        self.calls += 1
        return reserve

    def after(self, reserve, parsed):
        if not self.paid:
            return 0.0
        actual = parsed.get('actual_cost_usd')
        estimate = cost(self.model, parsed.get('input_tokens', 0) or 0, parsed.get('output_tokens', 0) or 0)
        charge = max(reserve, float(actual or 0), estimate or 0)
        self.spent += charge
        print(f'{self.model}: calls={self.calls}/{self.max_calls}, conservative running cost=${self.spent:.6f}', flush=True)
        return charge


def execute(client, task, item, path, budget, warmup=False, body=None, extra=None, parser=None):
    request = body if body is not None else client.body(task, item['text'])
    for attempt in range(4):  # initial attempt plus up to three transient retries
        reserve = budget.before(request)
        row = {'ts_utc': now(), 'model': budget.model, 'task': task.name, 'id': item['id'], 'label': item['label'],
               'request': request, 'response': None, 'latency_ms': None, 'input_tokens': 0, 'output_tokens': 0,
               'pred': None, 'probs': {}, 'valid': False, 'error': None, 'warmup': warmup, 'attempt': attempt + 1}
        row.update(extra or {})
        retry = False
        try:
            response, elapsed = client.call(request)
            row.update(response=response, latency_ms=elapsed)
            usage = response.get('usage') or {}
            row.update(input_tokens=usage.get('input_tokens', usage.get('prompt_tokens', 0)),
                       output_tokens=usage.get('output_tokens', usage.get('completion_tokens', 0)), actual_cost_usd=usage.get('cost'))
            row.update(parser(response) if parser else client.parse(response, task))
        except CallError as exc:
            row.update(response=exc.response, latency_ms=exc.latency_ms, error=str(exc))
            retry = exc.transient or exc.unsupported
            if exc.unsupported:
                client.text_only = True
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        row['charged_cost_usd'] = budget.after(reserve, row)
        append(path, row)
        if not row['error']:
            return row
        print(f'{budget.model}/{task.name}/{item["id"]}: {row["error"]}', flush=True)
        if not retry or attempt == 3:
            return row
        if getattr(client, 'text_only', False):
            request = client.body(task, item['text'])
        time.sleep(2 ** attempt)


def run_task(model, task, limit, max_calls, budget=None, client=None):
    items = read_jsonl(DATA / f'{task.name}.jsonl')
    if not items:
        raise ValueError(f'No data for {task.name}; run prepare_data.py first')
    if limit is not None:
        items = items[:limit]
    path = log_path(model, task.name)
    done = {str(r['id']) for r in successful(read_jsonl(path))}
    todo = [r for r in items if str(r['id']) not in done]
    if not todo:
        print(f'{model}/{task.name}: nothing to run (already complete).')
        return budget, client
    client = make_client(model, task) if client is None or model == 'tfidf-lr' else client
    budget = budget or Budget(model, client.paid, max_calls)
    execute(client, task, todo[0], path, budget, warmup=True)
    for i, item in enumerate(todo, 1):
        row = execute(client, task, item, path, budget)
        print(f'{model}/{task.name}: {i}/{len(todo)} pred={row["pred"]} label={item["label"]} latency={row["latency_ms"]} ms', flush=True)
    return budget, client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--task', choices=TASKS)
    group.add_argument('--all', action='store_true', help='Run all four tasks for the selected model')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-calls', type=int, default=1000)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    budget = client = None
    try:
        for name in TASKS if args.all else [args.task]:
            budget, client = run_task(args.model, TASKS[name], args.limit, args.max_calls, budget, client)
    except SkipModel as exc:
        print(exc)
    except (BudgetExceeded, ValueError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')

if __name__ == '__main__':
    main()
