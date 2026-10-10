"""Local-only shared-state experiment; randomized question counts per item."""
import argparse
import random
from common import DATA, append, log_path, read_jsonl
from models import HTTPClient, parse_decision
from run import Budget, BudgetExceeded, execute
from tasks import TASKS

QUESTIONS = {
    'team': TASKS['routing'].question(),
    'is_complaint': {'type': 'noul', 'instructions': 'Is this customer making a complaint?'},
    'urgency': {'type': 'score', 'instructions': 'How urgent is this customer request?',
                'criteria': ['Low: routine or no time pressure', 'Medium: needs timely attention', 'High: requires immediate attention']},
    'wants_refund': {'type': 'noul', 'instructions': 'Does this customer want a refund?'},
    'is_angry': {'type': 'noul', 'instructions': 'Is this customer angry?'},
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        parser.error('--limit must be between 1 and 30')
    items = read_jsonl(DATA / 'routing.jsonl')[:args.limit]
    if not items:
        parser.error('Run prepare_data.py first')
    client = HTTPClient('d1-local')
    budget = Budget('d1-local', False, 1000)
    path = log_path('d1-local', 'multiq')
    done = {(r['id'], r['k']) for r in read_jsonl(path) if not r.get('warmup') and not r.get('error')}
    rng = random.Random(42)
    warm = False
    for item in items:
        ks = [1, 2, 3, 5]
        rng.shuffle(ks)
        for k in ks:
            if (item['id'], k) in done:
                continue
            body = {'state': item['text'], 'questions': dict(list(QUESTIONS.items())[:k])}
            def parse(response):
                parsed = parse_decision(response, TASKS['routing'], 'team')
                answers = response.get('answers', response.get('results', response))
                if not set(body['questions']).issubset(answers):
                    raise ValueError('Response missing multi-question answers')
                return parsed
            if not warm:
                execute(client, TASKS['routing'], item, path, budget, True, body, {'task': 'multiq', 'k': k}, parse)
                warm = True
            row = execute(client, TASKS['routing'], item, path, budget, False, body, {'task': 'multiq', 'k': k}, parse)
            print(f'{item["id"]}: k={k}, latency={row["latency_ms"]:.1f}ms, tokens={row["input_tokens"]}', flush=True)

if __name__ == '__main__':
    main()
