# 002 - Colab GPU notebook v1: external code review findings

Source: code review of `colab/d1_gpu_speed.py` v1 shared by the author (10 Oct 2026). Fixes landed in commit `f19c7a2` ("GPU notebook: fixes from code review").

| # | Finding | Valid? | Fix |
|---|---|---|---|
| 1 | Script fails to compile (`\#!`, HTML entities) | No - artifact of copying into Markdown; the real file compiles | none |
| 2 | Accuracy computed over parsed answers only (hides invalid outputs) | Yes | Report `accuracy_on_parsed`, `end_to_end_accuracy`, `parse_rate`, `error_rate`, `malformed_rate` |
| 3 | Qwen parser exact-match only (`A)`, `A.` counted invalid) | Yes | Score Qwen by first-token logprobs like the CPU harness; narrow text parse as cross-check |
| 4 | d1 vs Qwen aren't computationally equivalent (no generation vs autoregressive) | Yes, framing | Record prompt tokens, prefill/eval timings, GPU memory; README says this is end-to-end application performance, not architecture efficiency |
| 5 | GPU offload check passed with partial offload | Yes | Require `offloaded N/N layers`; abort otherwise; log nvidia-smi memory per model |
| 6 | Multi-question throughput counted requests as decisions; wall time = sum of latencies | Yes | Real wall-clock; report requests/s and decisions/s |
| 7 | Fixed run order (d1 always first), single warmup | Yes | 2 reps with alternating order, 5 warmups per server start |
| 8 | Parse exceptions outside try could crash the run | Yes | Parsing inside error handling; buckets: http_errors / malformed / unparsed / parsed |
| 9 | No model integrity check; unused llama-cpp-python install | Yes | HF revisions pinned, SHA-256 verified; unused install removed |
| 10 | First-N sample for multi-question; `-np 4` for single-request latency; one flash_attn value for both | Yes | Seeded sample; `-np 1` for c=1, `-np 4` for c=4; per-model config recorded |

Reviewer's extra idea (not done yet): cost per 1M decisions from measured GPU throughput x GPU $/hour.
