import json
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = ROOT / 'results' / 'raw'

def now():
    return datetime.now(timezone.utc).isoformat()

def log_path(model, task):
    return RAW / f'{re.sub(r"[^a-zA-Z0-9_.:-]", "_", model)}__{task}.jsonl'

def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f'Warning: ignoring malformed line {path}:{i}')
    return rows

def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def successful(rows):
    # Last successful evaluation per id; retries and warmups are not extra samples.
    return list({str(r['id']): r for r in rows if not r.get('warmup') and not r.get('error')}.values())
