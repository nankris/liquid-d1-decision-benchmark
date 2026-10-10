#!/usr/bin/env bash
# Full local run: d1-3B vs Qwen2.5-3B-Instruct vs TF-IDF+LR, then multi-question scaling.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_DATASETS_OFFLINE=1
for m in d1-local qwen-local tfidf-lr; do
  python3 harness/run.py --model "$m" --all
done
python3 harness/multiq.py
python3 harness/analyze.py | tee results/summary_stdout.txt
python3 harness/charts.py
