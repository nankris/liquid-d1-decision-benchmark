# 005 - GPU: concurrency 4 gave no throughput gain (open question)

**Run:** Colab T4, third run (code at da59125), 10 Oct 2026. Data: `results/gpu_t4/d1_gpu_speed_results.json`.

**What we saw.** Concurrency 4 ran on a fresh `llama-server -np 4 -c 8192` (2048 ctx per slot), 4 client threads. Throughput stayed flat and latency went up roughly 4x:

| Model / task | req/s c=1 (pooled) | req/s c=4 | p50 c=1 -> c=4 |
|---|---|---|---|
| d1 spam | 27.0 | 23.6 (rep 2; pooled 11.9 includes the 13 s stall, see 003) | 35 -> 164 ms |
| d1 routing | 12.6 | 12.0 | 78 -> 333 ms |
| Qwen spam | 13.4 | 14.3 | 72 -> 280 ms |
| Qwen routing | 7.2 | 7.5 | 137 -> 527 ms |

**What the JSON tells us**
- Not a d1-endpoint quirk: Qwen on `/v1/chat/completions` shows the same flat throughput (+5-6%).
- Not plain request serialization: the first four requests of most c=4 batches finished at about the same time (e.g. d1 routing rep 1: 337/342/332/331 ms; Qwen spam rep 1: 282/283/294/286 ms), so llama-server put them in one batch. The batch just took ~4x as long as one request.
- Work per request is almost all prefill (69-224 prompt tokens; d1 does no generation, Qwen generates <=4 tokens). At c=1 d1 already processes ~1,850-2,200 prompt tokens/s, plausibly close to what a T4 can prefill for a 3B Q4 model (not measured here; see next steps). Prefill is compute-bound, so batching it adds little; only Qwen's few decode steps (memory-bound) can gain, which matches its small +5%.
- `-t 2 -tb 2` (Colab has 2 vCPUs) should matter little with all layers on the GPU, but the Python client threads, HTTP handling and Qwen's top-20 logprob sampling share those 2 vCPUs. Not ruled out.

**Gaps.** `mean_prompt_ms` / `predicted_ms` are null for both models: the responses carried no `timings` this run, and GPU utilization wasn't sampled. So "prefill-bound" is the best explanation, not a measured fact.

**To check next run (cheap):** sample `nvidia-smi --query-gpu=utilization.gpu` during c=1 and c=4 batches; `llama-bench -p 64,128,256 -b/-ub` on the T4 to get peak prefill tokens/s; log server per-request timings (or parse the server log); try c=4 with more client headroom.

**Impact on the write-up.** Report c=1 numbers as the headline; state that c=4 didn't raise throughput on a T4 for either model. Don't claim d1 scales with concurrency.
