# 003 - First Colab T4 run: prebuilt llama.cpp saw no CUDA device

**When:** 10 Oct 2026, ~5:30-5:50 PM IST. Account with Colab compute units, T4 runtime.

**Environment printed by the notebook:** `GPU Tesla T4  driver 580.82.07  CUDA 13.0  15360 MiB`

**What happened**
1. Downloaded `llama-b11541-bin-ubuntu-cuda-12.8-x64.tar.gz` (171,990,585 bytes) and extracted it.
2. `llama-server --list-devices` printed `Available devices: (none)`.
3. Script fell back to building b11541 from source for sm_75 with `cmake --build ... -j 2`.
4. Output stopped changing after ~2 minutes and stayed frozen ~18 minutes; Colab status went to "Connecting / Resuming execution".
5. Run aborted manually; runtime disconnected and deleted (Manage sessions: no active sessions). No results produced. ~20 min of T4 time used.

**Suspected cause (to confirm):** the prebuilt CUDA backend can't load on Colab - likely missing CUDA 12 runtime libraries (libcudart/libcublas 12) on Colab's CUDA 13 image, or backend libs not on `LD_LIBRARY_PATH`. Source build on Colab's 2 vCPUs is too slow to be a fallback.

**Fixes planned**
- Diagnose with `ldd` on the CUDA backend; ship missing CUDA 12 libs via pip `nvidia-*-cu12` wheels and set `LD_LIBRARY_PATH`; or use a CUDA 13 asset if one exists.
- Print `ldd` output when device detection fails.
- Fail fast instead of silently building from source; source build opt-in only (`BUILD_FROM_SOURCE=1`).

**Lesson:** test device detection in a 1-minute smoke cell before committing to the full run, so a failure costs ~1 minute of GPU, not 20.
