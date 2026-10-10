"""Build deterministic, source-row-disjoint evaluation and training files."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import json
import re
from collections import Counter
import numpy as np
from datasets import load_dataset
from common import DATA
from tasks import CATEGORIES

PLACEHOLDERS = {'order number': '#48213', 'invoice number': 'INV-48213', 'customer name': 'Alex Morgan',
                'name': 'Alex', 'email': 'alex@example.com', 'website': 'shop.example.com',
                'phone number': '202-555-0142', 'date': '2026-01-15', 'product name': 'wireless headphones'}

def replace_placeholders(text):
    return re.sub(r'\{\{\s*(.*?)\s*\}\}', lambda m: PLACEHOLDERS.get(m[1].strip().lower(), m[1].strip()), text)

def truncate(text):
    text = text.replace('\\n', '\n').replace('\\"', '"')
    if len(text) <= 1000:
        return text
    chunk = text[:999]
    return (chunk.rsplit(None, 1)[0] if any(c.isspace() for c in chunk) else chunk) + '…'

def balanced_indices(labels, counts, rng):
    labels = np.asarray(labels)
    selected = np.concatenate([rng.choice(np.flatnonzero(labels == label), count, replace=False) for label, count in counts.items()])
    rng.shuffle(selected)
    return selected.tolist()

def rows(ds, indices, split, text_field, transform=lambda x: x, label_field='label'):
    return [{'id': f'{split}:{i}', 'text': transform(ds[int(i)][text_field]), 'label': ds[int(i)][label_field]} for i in indices]

def write(task, evaluation, training):
    assert len(evaluation) == 200
    assert not {r['id'] for r in evaluation} & {r['id'] for r in training}
    for suffix, values in [('', evaluation), ('_train', training)]:
        (DATA / f'{task}{suffix}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
    print(f'{task}: eval={len(evaluation)}, train={len(training)}, labels={dict(Counter(r["label"] for r in evaluation))}')

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    ds = load_dataset('ucirvine/sms_spam', split='train')
    ix = balanced_indices(ds['label'], {0: 140, 1: 60}, rng)
    remaining = sorted(set(range(len(ds))) - set(ix))
    write('spam', rows(ds, ix, 'train', 'sms'), rows(ds, remaining, 'train', 'sms'))
    ds = load_dataset('cardiffnlp/tweet_eval', 'offensive')
    positive = round(200 * np.mean(ds['test']['label']))
    ix = balanced_indices(ds['test']['label'], {0: 200 - positive, 1: positive}, rng)
    write('toxic', rows(ds['test'], ix, 'test', 'text'), rows(ds['train'], range(len(ds['train'])), 'train', 'text'))
    ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset', split='train')
    assert set(ds['category']) == set(CATEGORIES), f'Unexpected routing labels: {set(ds["category"])}'
    counts = {label: 200 // len(CATEGORIES) + (i < 200 % len(CATEGORIES)) for i, label in enumerate(CATEGORIES)}
    ix = balanced_indices(ds['category'], counts, rng)
    rest = rng.choice(sorted(set(range(len(ds))) - set(ix)), 3000, replace=False).tolist()
    write('routing', rows(ds, ix, 'train', 'instruction', replace_placeholders, 'category'), rows(ds, rest, 'train', 'instruction', replace_placeholders, 'category'))
    ds = load_dataset('Yelp/yelp_review_full')
    ix = balanced_indices(ds['test']['label'], dict.fromkeys(range(5), 40), rng)
    rest = rng.choice(len(ds['train']), 5000, replace=False).tolist()
    write('rating', rows(ds['test'], ix, 'test', 'text', truncate), rows(ds['train'], rest, 'train', 'text', truncate))

if __name__ == '__main__':
    main()
