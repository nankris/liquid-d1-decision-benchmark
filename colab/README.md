# Colab T4 speed test

About 20–25 minutes on a Colab T4. Compares Liquid d1-3B and Qwen2.5-3B-Instruct, both Q4_K_M, on the same 200 spam items and 200 routing items.

## Run

1. Upload `d1_gpu_speed.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. `d1data.zip` is downloaded from this repo's raw URL into `/content/d1data.zip` if it is not already there (you can still upload it by hand).

The main cell is `d1_gpu_speed.py`. It runs the timed calls through **llama.cpp `llama-server` b11541** (prebuilt Ubuntu CUDA 12.8 build plus its `cudart-llama` runtime archive; no source build unless you set `BUILD_FROM_SOURCE=1`). d1 uses `POST /v1/systemone`. Qwen uses `POST /v1/chat/completions` with exactly the CPU harness request body (`temperature=0`, `max_tokens=4`, `logprobs=true`, `top_logprobs=20`).

The last cell disconnects the runtime. The summary table and the JSON are printed in the cell output. The file is `/content/d1_gpu_speed_results.json` (it also holds raw per-request latencies). Download it from the files pane before that last cell if you want the file itself.

`d1_gpu_speed.py` is the same program. From a Colab cell, after uploading it next to the notebook: `%run d1_gpu_speed.py`.

## What this measures (and what it does not)

This is **end-to-end application performance** as served by llama-server: HTTP latency, throughput and accuracy for each model behind its own endpoint. It is **not** a measure of architecture efficiency. The two endpoints build different prompts and do different server-side work, so the JSON records the context needed to interpret the numbers: mean prompt tokens per request, llama-server `timings` (prefill/eval ms and token counts) when the response includes them, flash-attention setting, layers offloaded and GPU memory per model.

## Protocol

- **Models** are pinned to Hugging Face commits (`/resolve/<sha>/`) and checked against the LFS SHA-256 from the HF API before use: `LiquidAI/d1-3B-GGUF@bb1e436e…`, `Qwen/Qwen2.5-3B-Instruct-GGUF@7dabda4d…`.
- **Full GPU offload is required.** Every server start must log `offloaded N/N layers to GPU` with N equal to the total; otherwise the run aborts. `nvidia-smi` memory used after load and after the runs is recorded per model and server start.
- **Concurrency 1** runs on a server started with `-np 1`; **concurrency 4** on a freshly restarted server with `-np 4`. Context is 2048 tokens per slot in both. Prompt cache is off.
- **Warmup:** 5 unmeasured requests per server start, at the run's concurrency.
- **Order bias:** two repetitions with alternating model order.
  - Rep 1: d1 then Qwen, all 200 spam + 200 routing items (accuracy is reported from this rep).
  - Rep 2: Qwen then d1, a seeded 100-item subset per task (`random.Random(44)`), mainly for latency. It drops to 50 items if more than 14 minutes have elapsed, and is skipped after 19 minutes; the JSON says which.
  - The JSON reports per-rep rows and pooled rows (latency percentiles over all requests from both reps; throughput = total successes / total wall time).
- **Throughput** is successful requests divided by real elapsed wall-clock time for the batch (`time.perf_counter` around the whole batch), reported as `throughput_requests_per_s` and `throughput_decisions_per_s` (= requests × k / wall; k = 1 for spam/routing).
- **Multi-question test** (d1 only, rep 1, concurrency 1): a seeded random sample of 30 routing items (`random.Random(42).sample`; 10 if the run is already past 12 minutes), k in {1, 2, 3, 5}, questions `team`, `is_complaint`, `urgency`, `wants_refund`, `is_angry`. Each k is its own timed batch (k order shuffled with the same seed). Note this differs from `harness/multiq.py`, which uses the first 30 items interleaved per item.

### Accuracy and error accounting

Every request lands in exactly one bucket: `http_errors` (transport failure or non-2xx), `malformed` (2xx but the body could not be parsed: not JSON, missing fields, bad numbers), `unparsed` (well-formed but no valid answer), or `parsed`. Parsing runs inside the same error handling as the request. Reported per row:

- `accuracy_on_parsed` = correct / parsed
- `end_to_end_accuracy` = correct / n (errors and invalid answers count as wrong)
- `parse_rate` = parsed / n, `error_rate` = http_errors / n, `malformed_rate` = malformed / n

### How Qwen's answer is read

No grammar or `logit_bias` is used, so the Qwen request stays byte-for-byte the CPU harness request and CPU and GPU results stay comparable. The prediction uses the CPU harness rule (`harness/models.py`): sum the first-token `top_logprobs` mass for each valid answer token (stripped, lowercased) and take the argmax. With single-token answers this is equivalent to constraining the first generated token to the valid set. As a cross-check the JSON also reports a text parse: strict exact match, and a narrow normalization that accepts the token with whitespace, one trailing `)`, `.` or `:`, and optionally the option's own label (`A`, `A.`, `A) ACCOUNT`, `Yes.`; a label prefix only when cut off by `max_tokens`). A letter anywhere in free text is not accepted. If the server returns no logprobs, the normalized text parse is used.

### Recorded in the JSON

Dataset SHA-256 (zip and each jsonl), item and label counts; model repo, revision, SHA-256, size; per-server command, flash-attention setting (each model's own, since `-fa on` falls back to `off` if the server fails to start), offloaded layers, GPU memory; mean prompt tokens and llama-server timings per row.

Routing labels use **SUBSCRIPTION**, not NEWSLETTER.

## Changes from code review (2026-10)

- Accuracy split into `accuracy_on_parsed` and `end_to_end_accuracy`, plus `parse_rate`, `error_rate` and separate `http_errors` / `malformed` / `unparsed` / `parsed` counters; parsing moved inside the request error handling.
- Qwen scored with the CPU harness logprob rule (request now includes `logprobs` / `top_logprobs` like the harness); narrow normalized text parse reported alongside. Mean prompt tokens and llama-server timings recorded.
- Full GPU offload enforced from the server log; GPU memory recorded per model and server start.
- Multi-question throughput now uses real wall-clock per batch and reports requests/s and decisions/s; it samples items with a fixed seed instead of taking the first N.
- Two repetitions with alternating model order, 5 warmups per server start, per-rep and pooled stats.
- `-np 1` for concurrency 1 and `-np 4` for concurrency 4 (server restarted between).
- Hugging Face downloads pinned to commit SHAs and SHA-256 verified. The unused `llama-cpp-python` install was removed (only `llama-server` is used).
- Per-model flash-attention setting, dataset SHA-256 and counts in the report.

## If the prebuilt CUDA server does not start

How the GPU path works: the b11541 Ubuntu CUDA tarball ships the CUDA backend as a separate `libggml-cuda.so` that llama-server `dlopen()`s from its own directory at startup. That library links `libcudart.so.12`, `libcublas.so.12` (and through it `libcublasLt.so.12`), which are **not** in the binary tarball; they come in the companion `cudart-llama-b11541-bin-ubuntu-cuda-12.8-x64.tar.gz`. Without them ggml skips the backend silently and `llama-server --list-devices` prints `(none)`. `ldd llama-server` does not reveal this because the server binary does not link the backend. The script therefore always downloads the cudart archive, links its libraries next to `libggml-cuda.so` (whose RUNPATH is `$ORIGIN`), adds them, the NVIDIA driver directory (`/usr/lib64-nvidia` on Colab) and, if anything is still unresolved, NVIDIA's `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` pip wheels (installed into a private `--target` dir) to `LD_LIBRARY_PATH`, and runs the server with its directory as cwd.

The 12.8 build is used rather than the 13.4 one even though Colab's driver reports CUDA 13.0: neither build has T4 (sm_75) machine code, only compute_75 PTX that the driver JIT-compiles, and the 13.4 build's PTX (ISA 9.4) is newer than a CUDA 13.0 driver can compile. The 12.8 PTX (ISA 8.7) is fine. The first server start therefore spends some extra time JIT-compiling kernels (the cache is kept in `/content/llama/cuda-jit-cache`), and the first load and warmup requests get longer timeouts. The 13.4 build is only tried when the driver reports CUDA 13.4 or newer; `LLAMA_CUDA_FLAVOR=12.8|13.4` forces one.

If no CUDA device shows up, the script prints diagnostics (the `--list-devices` output, `LD_LIBRARY_PATH`, where `libcuda.so.1` was found, `ldd libggml-cuda.so`, and the actual `dlopen` error) and exits without measuring. It does not build from source on its own: on Colab's 2 vCPUs that takes 20+ minutes. To opt in, run `import os; os.environ["BUILD_FROM_SOURCE"] = "1"` in a cell before the main cell. The build uses `-j $(nproc)`, prints cmake progress, and is killed after `BUILD_TIMEOUT_MIN` minutes (default 45) with a clear message; the full log is `/content/llama/build.log`.
