# Colab T4 speed test

About 20 minutes on a Colab T4. Compares Liquid d1-3B and Qwen2.5-3B-Instruct, both Q4_K_M, on the same 200 spam items and 200 routing items.

## Run

1. Upload `d1_gpu_speed.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU**.
3. Upload `d1data.zip` to `/content/d1data.zip` (Files pane, into `/content`, not a subfolder).
4. Runtime → Run all.

The main cell is `d1_gpu_speed.py`. It installs a prebuilt `llama-cpp-python` CUDA wheel when one matches this machine, then runs the timed calls through **llama.cpp `llama-server` b11541** (CUDA 12.8 build, or a source build if that binary cannot see the GPU). d1 uses `POST /v1/systemone`. Qwen uses `POST /v1/chat/completions` with the harness prompts (`temperature=0`, `max_tokens=4`, no logprobs).

The last cell disconnects the runtime. The summary table and the full JSON are printed in the cell output. The file is `/content/d1_gpu_speed_results.json`. Download it from the files pane before that last cell if you want the file itself.

`d1_gpu_speed.py` is the same program. From a Colab cell, after uploading it next to the notebook: `%run d1_gpu_speed.py`.

## What is measured

For each model and each task, at parallel 1 and parallel 4:

- latency p50 and p95 (milliseconds, nearest-rank, HTTP time)
- throughput in decisions per second (successes / wall clock)

Parallel 4 is four client requests at a time against one server started with `-np 4` and `-ngl all`. Parallel 1 is one request at a time on that same server. One warmup call per model is excluded. Prompt cache is off.

d1 also runs the multi-question check from `harness/multiq.py`: the first 30 routing items, k in {1, 2, 3, 5}, questions `team`, `is_complaint`, `urgency`, `wants_refund`, `is_angry`. If the run is already past 15 minutes, that check uses 10 items instead. The JSON records which size ran.

Routing labels use **SUBSCRIPTION**, not NEWSLETTER.

## If the prebuilt CUDA server does not start

The notebook prints a source-build fallback. For the Python binding only:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install llama-cpp-python --no-binary llama-cpp-python
```

That compile is often longer than 10 minutes. The timed path prefers the prebuilt `llama-server` because it exposes `/v1/systemone`.
