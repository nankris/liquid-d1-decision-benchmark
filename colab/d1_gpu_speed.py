#!/usr/bin/env python3
"""Colab T4 speed test: Liquid d1-3B vs Qwen2.5-3B-Instruct, both Q4_K_M.

Same spam and routing prompts as harness/tasks.py (the routing label is
SUBSCRIPTION, not NEWSLETTER) and the same multi-question set as
harness/multiq.py. Measures end-to-end HTTP latency, throughput and
accuracy at concurrency 1 and 4.

This measures application-level performance of each model as served by
llama-server (d1's /v1/systemone vs Qwen's chat completions), not the
efficiency of either architecture in isolation. The two requests differ
in prompt length and in server-side work.

Inference is llama.cpp llama-server b11541 with CUDA. Every server start
must report full offload ("offloaded N/N layers to GPU") or the run
aborts. Concurrency 1 runs on a server started with -np 1, concurrency 4
on a server restarted with -np 4 (2048 context tokens per slot in both).
d1 is POST /v1/systemone. Qwen is POST /v1/chat/completions with the CPU
harness body (temperature 0, max_tokens 4, logprobs with top_logprobs 20);
its prediction is scored exactly like harness/models.py (first-token
probability mass over the valid answer tokens), with a normalized text
parse as a cross-check. Prompt cache is off so each call pays a full
prefill.

Two repetitions alternate model order (rep 1: d1 then Qwen on all 200
items per task; rep 2: Qwen then d1 on a seeded 100-item subset per task).
Each server start gets 5 warmup requests that are not measured.

The notebook colab/d1_gpu_speed.ipynb embeds this file as its main cell.
Its last cell disconnects the Colab runtime. This script does not.

colab/d1data.zip is downloaded from the GitHub repo into /content when it
is not already there.
"""

import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


LLAMA_TAG = "b11541"
LLAMA_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
# CUDA 12.8 build runs on the Colab driver that advertises CUDA 12.8+ (the
# April 2026 T4 image reports driver 580 / CUDA 13.0). There is no Ubuntu
# CUDA 12.4 binary in this release. The companion cudart archive is only
# downloaded when the system CUDA libraries are not enough.
CUDA_BIN_URL = f"{LLAMA_RELEASE}/llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
CUDA_LIBS_URL = f"{LLAMA_RELEASE}/cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"

# Hugging Face files pinned to a commit. sha256 and size are the LFS values
# from https://huggingface.co/api/models/<repo>?blobs=true at that commit
# (resolved 2026-10-10). Downloads are verified against them.
MODELS = {
    "d1-3B": {
        "repo": "LiquidAI/d1-3B-GGUF",
        "revision": "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563",
        "file": "d1-3B-Q4_K_M.gguf",
        "sha256": "16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402",
        "bytes": 1674456672,
        "alias": "d1-3B",
    },
    "qwen2.5-3b-instruct": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "sha256": "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
        "bytes": 2104932768,
        "alias": "Qwen2.5-3B-Instruct",
    },
}
for _spec in MODELS.values():
    _spec["url"] = f"https://huggingface.co/{_spec['repo']}/resolve/{_spec['revision']}/{_spec['file']}"
QWEN_MODEL_ID = MODELS["qwen2.5-3b-instruct"]["alias"]
KIND_TO_MODEL = {"d1": "d1-3B", "qwen": "qwen2.5-3b-instruct"}

DATA_ZIP_URL = "https://raw.githubusercontent.com/nankris/liquid-d1-decision-benchmark/main/colab/d1data.zip"

HOST = "127.0.0.1"
PORT = 8081
CTX_PER_SLOT = 2048  # -c is CTX_PER_SLOT * -np, so each slot gets the same context
CONCURRENCIES = (1, 4)  # server is started with -np equal to the concurrency
QWEN_MAX_TOKENS = 4  # same as harness/models.py
QWEN_TOP_LOGPROBS = 20  # same as harness/models.py
WARMUP_REQUESTS = 5
SEED = 42
REP2_ITEMS = 100
REP2_ITEMS_SHORT = 50
REP2_SHRINK_AFTER_S = 14 * 60
REP2_SKIP_AFTER_S = 19 * 60
MULTIQ_FULL = 30
MULTIQ_SHORT = 10
MULTIQ_CUTOFF_S = 12 * 60
MULTIQ_KS = (1, 2, 3, 5)

# Copied verbatim from harness/tasks.py. Do not swap SUBSCRIPTION for NEWSLETTER.
CATEGORIES = {
    "ACCOUNT": "creating, deleting, editing or switching an account, password recovery, registration problems",
    "CANCEL": "questions about cancellation or early-termination fees/charges",
    "CONTACT": "wants to reach customer service or talk to a human agent",
    "DELIVERY": "delivery options and how long delivery takes",
    "FEEDBACK": "complaints, formal claims, or leaving a review",
    "INVOICE": "viewing, finding or getting a copy of an invoice or bill",
    "SUBSCRIPTION": "subscribing to or unsubscribing from the newsletter",
    "ORDER": "placing, changing, cancelling or tracking an order",
    "PAYMENT": "accepted payment methods or problems making a payment",
    "REFUND": "refund policy, requesting a refund, or checking refund status",
    "SHIPPING": "setting up or changing the shipping address",
}
SYSTEM = "You are a careful classifier. Reply with only the answer token, nothing else."


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    instructions: str
    criteria: object = None

    @property
    def options(self):
        if self.kind == "noul":
            return ["0", "1"]
        if self.kind == "choice":
            return list(self.criteria)
        return [str(i) for i in range(len(self.criteria))]

    @property
    def tokens(self):
        if self.kind == "noul":
            return {"no": "0", "yes": "1"}
        if self.kind == "choice":
            return {chr(65 + i): label for i, label in enumerate(self.options)}
        return {str(i + 1): str(i) for i in range(len(self.criteria))}

    def question(self):
        q = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            q["criteria"] = self.criteria
        return q

    def messages(self, text):
        if self.kind == "noul":
            options = "yes) yes\nno) no"
        elif self.kind == "choice":
            options = "\n".join(
                f"{token}) {label} — {self.criteria[label]}" for token, label in self.tokens.items()
            )
        else:
            options = "\n".join(f"{i + 1}) {description}" for i, description in enumerate(self.criteria))
        user = f"{self.instructions}\n{options}\n\nText:\n{text}\n\nAnswer:"
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


TASKS = {
    "spam": Task(
        "spam",
        "noul",
        "Is this SMS message spam (unsolicited advertising, scams, prize or premium-rate number promotions)?",
    ),
    "routing": Task(
        "routing",
        "choice",
        "Which support team should handle this customer message?",
        CATEGORIES,
    ),
}

# Copied from harness/multiq.py. `team` is the routing question; k uses this order.
QUESTIONS = {
    "team": TASKS["routing"].question(),
    "is_complaint": {"type": "noul", "instructions": "Is this customer making a complaint?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this customer request?",
        "criteria": [
            "Low: routine or no time pressure",
            "Medium: needs timely attention",
            "High: requires immediate attention",
        ],
    },
    "wants_refund": {"type": "noul", "instructions": "Does this customer want a refund?"},
    "is_angry": {"type": "noul", "instructions": "Is this customer angry?"},
}


def on_colab():
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return Path("/content/sample_data").is_dir()


def workspace():
    if on_colab():
        root = Path("/content")
    else:
        root = Path(os.environ.get("D1_SPEED_ROOT", "/tmp/d1_gpu_speed"))
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "models": root / "models",
        "llama": root / "llama",
        "downloads": root / "llama" / "downloads",
        "results": root / "d1_gpu_speed_results.json",
        "log": root / "llama-server.log",
    }


def percentile(values, p):
    """Nearest-rank percentile. p is in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((p / 100) * len(ordered)))
    return ordered[rank - 1]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def check(cmd, **kwargs):
    result = run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def remote_size(url):
    result = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True)
    size = None
    for line in (result.stdout or "").splitlines():
        if line.lower().startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
    return size


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(url, dest: Path, min_bytes=1):
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    if dest.exists() and dest.stat().st_size >= min_bytes and (not expected or dest.stat().st_size == expected):
        print(f"cached {dest} ({dest.stat().st_size} bytes)", flush=True)
        return dest
    if dest.exists():
        dest.unlink()
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists() and expected and partial.stat().st_size == expected:
        partial.replace(dest)
        print(f"cached partial {dest.name} ({dest.stat().st_size} bytes)", flush=True)
        return dest
    if partial.exists() and expected and partial.stat().st_size > expected:
        partial.unlink()
    check(
        [
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2",
            "-C", "-", "-A", "d1-gpu-speed/1.0", "-o", str(partial), url,
        ]
    )
    got = partial.stat().st_size
    if expected and got != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download size mismatch for {url}: got {got}, expected {expected}")
    if got < min_bytes:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download too small: {url}")
    partial.replace(dest)
    print(f"downloaded {dest.name} ({dest.stat().st_size} bytes)", flush=True)
    return dest


def extract_tarball(url, dest: Path):
    marker = dest / ".extracted"
    server = next(dest.rglob("llama-server"), None) if dest.exists() else None
    if marker.exists() and server and server.is_file():
        print(f"cached extract {dest}", flush=True)
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = download(url, workspace()["downloads"] / Path(url).name, min_bytes=1_000_000)
    check(["tar", "-xzf", str(archive), "-C", str(dest)])
    marker.write_text(url + "\n", encoding="utf-8")
    return dest


def find_llama_server(root: Path):
    matches = sorted(p for p in root.rglob("llama-server") if p.is_file() and os.access(p, os.X_OK))
    if not matches:
        matches = sorted(p for p in root.rglob("llama-server") if p.is_file())
    if not matches:
        raise RuntimeError(f"llama-server not found under {root}")
    return matches[0]


def so_dirs(root: Path):
    found = set()
    if not root.exists():
        return []
    for path in root.rglob("*"):
        if path.is_file() and ".so" in path.name:
            found.add(path.parent)
    return sorted(found)


def missing_libs(binary: Path, env):
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, env=env)
    text = (result.stdout or "") + (result.stderr or "")
    return [line.strip() for line in text.splitlines() if "not found" in line]


def make_env(lib_dirs):
    env = os.environ.copy()
    parts = []
    for directory in lib_dirs:
        if directory and Path(directory).is_dir() and str(directory) not in parts:
            parts.append(str(directory))
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def gpu_query():
    try:
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi is not available. Use a Colab runtime with a T4 GPU.") from exc
    text = (header.stdout or "") + (header.stderr or "")
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", text)
    if header.returncode != 0 or not match:
        raise RuntimeError("No NVIDIA GPU detected. In Colab: Runtime → Change runtime type → T4 GPU.\n" + text[-2000:])
    cuda = (int(match.group(1)), int(match.group(2)))
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    name, driver, memory = [part.strip() for part in rows[0].split(",")]
    info = {"name": name, "driver": driver, "cuda": f"{cuda[0]}.{cuda[1]}", "memory": memory}
    print(f"GPU {name}  driver {driver}  CUDA {info['cuda']}  {memory}", flush=True)
    if "T4" not in name:
        print(f"warning: this speed test is aimed at a T4; continuing on {name}", flush=True)
    return cuda, info


def gpu_mem_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])



def list_devices(binary, env):
    try:
        result = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True, text=True, env=env, timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        return False, (exc.stdout or "") + (exc.stderr or "") + "\n(timeout)"
    text = (result.stdout or "") + (result.stderr or "")
    ok = result.returncode == 0 and "CUDA" in text.upper() and "(none)" not in text
    return ok, text


def build_llama_server(cuda_arch="75"):
    """Source build pinned to llama.cpp b11541 for sm_75 (T4). Slow fallback."""
    paths = workspace()
    src = paths["llama"] / f"src-{LLAMA_TAG}"
    build = src / "build-cuda"
    binary = build / "bin" / "llama-server"
    env = make_env(so_dirs(build))
    if binary.exists():
        ok, _text = list_devices(binary, env)
        if ok:
            print(f"reusing source-built {binary}", flush=True)
            return binary, env
    print(
        "Prebuilt CUDA llama-server cannot see this GPU. Building b11541 from source "
        "for sm_75. This often takes 10 minutes and can push the run past 20.",
        flush=True,
    )
    if not src.exists():
        check(["git", "clone", "--depth", "1", "--branch", LLAMA_TAG, "https://github.com/ggml-org/llama.cpp", str(src)])
    jobs = str(max(1, os.cpu_count() or 2))
    check(
        [
            "cmake", "-B", str(build), "-S", str(src),
            "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    check(["cmake", "--build", str(build), "-j", jobs, "--target", "llama-server"])
    if not binary.exists():
        found = find_llama_server(build)
        binary = found
    ok, text = list_devices(binary, env)
    if not ok:
        raise RuntimeError("Built llama-server still has no CUDA device.\n" + text[-2000:])
    return binary, env


def ensure_llama_server(cuda):
    paths = workspace()
    if cuda < (12, 8):
        print(
            f"Driver CUDA is {cuda[0]}.{cuda[1]}, older than the b11541 Ubuntu CUDA 12.8 binary. "
            "Building from source against this machine's toolkit.",
            flush=True,
        )
        return build_llama_server()
    tree = extract_tarball(CUDA_BIN_URL, paths["llama"] / "cuda-12.8")
    binary = find_llama_server(tree)
    lib_dirs = [binary.parent, *so_dirs(tree)]
    env = make_env(lib_dirs)
    missing = missing_libs(binary, env)
    if missing:
        print("CUDA runtime libs missing from the binary tarball; fetching the cudart archive.", flush=True)
        print("\n".join(missing[:12]), flush=True)
        libs = extract_tarball(CUDA_LIBS_URL, paths["llama"] / "cudart-12.8")
        lib_dirs = [binary.parent, *so_dirs(tree), *so_dirs(libs)]
        env = make_env(lib_dirs)
        missing = missing_libs(binary, env)
        if missing:
            print("Still missing after cudart:\n" + "\n".join(missing[:12]), flush=True)
            return build_llama_server()
    ok, text = list_devices(binary, env)
    if not ok:
        print("Prebuilt llama-server --list-devices did not show CUDA:", flush=True)
        print(text[-1500:], flush=True)
        return build_llama_server()
    print(text.strip(), flush=True)
    return binary, env


def locate_zip():
    if on_colab():
        path = Path("/content/d1data.zip")
        if not path.is_file():
            import urllib.request
            try:
                print(f"downloading {DATA_ZIP_URL}", flush=True)
                urllib.request.urlretrieve(DATA_ZIP_URL, path)
            except Exception as exc:
                print("auto-download failed:", exc)
        if not path.is_file():
            raise RuntimeError(
                "Upload colab/d1data.zip to /content/d1data.zip "
                "(Files pane → upload into /content), then run again."
            )
        return path
    candidates = []
    if os.environ.get("D1_DATA_ZIP"):
        candidates.append(Path(os.environ["D1_DATA_ZIP"]))
    candidates.append(Path(__file__).resolve().parent / "d1data.zip")
    candidates.append(Path.cwd() / "colab" / "d1data.zip")
    candidates.append(Path.cwd() / "d1data.zip")
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("d1data.zip not found. On Colab, upload it to /content/d1data.zip.")


def extract_eval_zip(zip_path: Path, dest: Path):
    """Pull spam.jsonl and routing.jsonl out of the uploaded zip. Members may sit at the root."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for want in ("spam.jsonl", "routing.jsonl"):
            if want in names:
                member = want
            else:
                matches = [name for name in names if Path(name).name == want]
                if len(matches) != 1:
                    raise RuntimeError(f"{zip_path} does not contain a single {want} (found {matches})")
                member = matches[0]
            target = dest / want
            with archive.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    rows = {}
    meta = {"zip": {"path": str(zip_path), "sha256": sha256_file(zip_path), "bytes": zip_path.stat().st_size}}
    for want in ("spam.jsonl", "routing.jsonl"):
        data = read_jsonl(dest / want)
        if not data or not {"id", "text", "label"} <= set(data[0]):
            raise RuntimeError(f"{want} is empty or missing id/text/label")
        rows[want] = data
        labels = {}
        for row in data:
            labels[str(row["label"])] = labels.get(str(row["label"]), 0) + 1
        meta[want] = {
            "sha256": sha256_file(dest / want),
            "n_items": len(data),
            "label_counts": dict(sorted(labels.items())),
        }
        print(f"{want}: {len(data)} items sha256 {meta[want]['sha256'][:12]}", flush=True)
    return rows["spam.jsonl"], rows["routing.jsonl"], meta


def server_command(binary, model_path, alias, flash, n_slots):
    threads = str(min(8, os.cpu_count() or 2))
    return [
        str(binary),
        "-m", str(model_path),
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", "all",
        "-np", str(n_slots),
        "-c", str(CTX_PER_SLOT * n_slots),
        "-t", threads,
        "-tb", threads,
        "--fit", "off",
        "--no-mmproj",
        "--no-cache-prompt",
        "--slot-prompt-similarity", "0",
        "--reasoning", "off",
        "--no-reasoning-preserve",
        "--alias", alias,
        "--no-ui",
        "-fa", flash,
    ]


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def log_text(path: Path, tail=12000):
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data if tail is None else data[-tail:]


def wait_until_loaded(proc, log_path, timeout=180):
    """Readiness is HTTP /health. The C++ server block-buffers stdout when it is
    redirected to a file, so the log is only a secondary signal (stdbuf -oL)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}\n" + log_text(log_path)[-2500:])
        try:
            response = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if response.status_code == 200:
                time.sleep(0.3)
                return log_text(log_path)
        except requests.RequestException:
            pass
        text = log_text(log_path)
        if "model loaded" in text and "listening on" in text:
            return text
        time.sleep(0.4)
    raise RuntimeError("llama-server did not become ready\n" + log_text(log_path)[-2500:])


OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.IGNORECASE)


def check_full_offload(log_path, timeout=10):
    """Return (offloaded, total) from the llama-server log. Raises unless every layer is on the GPU."""
    deadline = time.time() + timeout
    match = None
    while time.time() < deadline:
        matches = OFFLOAD_RE.findall(log_text(log_path, tail=None))
        if matches:
            match = matches[-1]
            break
        time.sleep(0.5)
    if not match:
        raise RuntimeError(
            "llama-server log has no 'offloaded N/N layers to GPU' line, so full GPU offload "
            "cannot be verified. Aborting.\n" + log_text(log_path)[-2500:]
        )
    offloaded, total = int(match[0]), int(match[1])
    if offloaded != total or total == 0:
        raise RuntimeError(
            f"only {offloaded}/{total} layers are on the GPU; this benchmark requires full offload. Aborting."
        )
    return offloaded, total


def start_server(binary, env, model_path, alias, n_slots):
    paths = workspace()
    log_path = paths["log"]
    last_error = None
    for flash in ("on", "off"):
        if log_path.exists():
            log_path.unlink()
        cmd = server_command(binary, model_path, alias, flash, n_slots)
        print("starting server:", " ".join(cmd), flush=True)
        log_handle = open(log_path, "w", encoding="utf-8")
        launcher = ["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else []
        mem_before = gpu_mem_mib()
        proc = subprocess.Popen(
            launcher + cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            wait_until_loaded(proc, log_path)
        except Exception as exc:  # noqa: BLE001 — retry once with flash attention off
            last_error = exc
            stop_server(proc)
            log_handle.close()
            print(f"start with -fa {flash} failed: {exc}", flush=True)
            continue
        try:
            offloaded, total = check_full_offload(log_path)
        except Exception:
            stop_server(proc)
            log_handle.close()
            raise
        mem_after = gpu_mem_mib()
        print(
            f"server ready (-fa {flash}, -np {n_slots}), offloaded {offloaded}/{total} layers, "
            f"GPU memory used {mem_after} MiB (+{mem_after - mem_before} MiB)",
            flush=True,
        )
        return {
            "proc": proc,
            "log": log_handle,
            "cmd": cmd,
            "flash_attn": flash,
            "n_slots": n_slots,
            "ctx_size_total": CTX_PER_SLOT * n_slots,
            "offloaded_layers": offloaded,
            "total_layers": total,
            "gpu_mem_before_mib": mem_before,
            "gpu_mem_used_after_load_mib": mem_after,
            "gpu_mem_delta_mib": mem_after - mem_before,
        }
    raise RuntimeError(f"could not start llama-server for {model_path}") from last_error


def close_server(server):
    if not server:
        return
    stop_server(server["proc"])
    if server.get("log"):
        server["log"].close()


def server_record(server):
    return {key: value for key, value in server.items() if key not in ("proc", "log")}


def base_url():
    return f"http://{HOST}:{PORT}"


_thread_local = threading.local()


def http_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


class HttpError(RuntimeError):
    """Transport failure or non-2xx status."""


class MalformedResponse(ValueError):
    """2xx response whose body does not have the expected structure."""


def post_json(url, body):
    """Returns (payload, elapsed_ms). A 2xx body that is not JSON comes back as
    payload None so the parser counts it as malformed, not as an HTTP error."""
    started = time.perf_counter()
    try:
        response = http_session().post(url, json=body, timeout=(10, 60))
    except requests.RequestException as exc:
        raise HttpError(str(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000
    if not response.ok:
        raise HttpError(f"HTTP {response.status_code}: {response.text[:400]}")
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return payload, elapsed


def usage_info(payload):
    """Prompt tokens and llama-server timings, when the response carries them."""
    usage = payload.get("usage") or {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    if prompt_tokens is None:
        prompt_tokens = timings.get("prompt_n")
    info = {"prompt_tokens": prompt_tokens}
    for key in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms"):
        if timings.get(key) is not None:
            info[key] = timings[key]
    return info


def d1_answers(payload):
    if not isinstance(payload, dict):
        raise MalformedResponse("response is not a JSON object")
    answers = payload.get("answers", payload.get("results", payload))
    if not isinstance(answers, dict):
        raise MalformedResponse("answers is not an object")
    return answers


def parse_d1(payload, task, name="answer"):
    """Returns (pred or None, info). Raises MalformedResponse on a bad structure."""
    answers = d1_answers(payload)
    if name not in answers or not isinstance(answers[name], dict):
        raise MalformedResponse(f"missing answer {name!r}")
    answer = answers[name]
    info = usage_info(payload)
    if task.kind == "noul":
        try:
            p_yes = float(answer["noul"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedResponse("noul missing or not a number") from exc
        if not (math.isfinite(p_yes) and 0 <= p_yes <= 1):
            raise MalformedResponse(f"noul outside [0, 1]: {p_yes}")
        values = {"0": 1 - p_yes, "1": p_yes}
        return max(values, key=values.get), info
    probabilities = answer.get("probabilities") or {}
    if probabilities:
        try:
            pred = max(probabilities, key=lambda key: float(probabilities[key]))
        except (TypeError, ValueError) as exc:
            raise MalformedResponse("non-numeric probabilities") from exc
    else:
        pred = answer.get("choice")
    return (pred if pred in task.options else None), info


def normalize_text_answer(content, task, truncated=False):
    """Narrow normalization of a chat answer. Accepts the bare token with optional
    surrounding whitespace, one trailing ')', '.' or ':', and optionally the option's
    own label after it ('A', 'A.', 'A)', 'A: ACCOUNT', 'A) ACCOUNT.', 'Yes.').
    The label must belong to the same option; when the reply was cut off by
    max_tokens it may be a prefix of that label ('A) ACC'). Anything else is None.
    """
    mapping = {token.lower(): label for token, label in task.tokens.items()}
    text = (content or "").strip()
    match = re.fullmatch(r"([A-Za-z0-9]+)\s*[\).:]?(?:\s+([A-Za-z_]+)\s*\.?)?", text)
    if not match:
        return None
    token = match.group(1).lower()
    if token not in mapping:
        return None
    word = match.group(2)
    if word is None:
        return mapping[token]
    if task.kind == "noul":
        expected = token  # options read "yes) yes" / "no) no"
    elif task.kind == "choice":
        expected = mapping[token]
    else:
        return None
    word = word.lower()
    expected = expected.lower()
    if word == expected or (truncated and len(word) >= 1 and expected.startswith(word)):
        return mapping[token]
    return None


def parse_qwen(payload, task):
    """Returns (pred or None, info). Primary prediction is the CPU harness rule
    (harness/models.py parse_chat): sum first-token top_logprobs mass per valid
    answer token (token stripped and lowercased) and take the argmax. If the server
    returned no logprobs it falls back to the normalized text parse."""
    if not isinstance(payload, dict):
        raise MalformedResponse("response is not a JSON object")
    try:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise MalformedResponse("no choices[0].message") from exc
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise MalformedResponse("content is not a string")
    mapping = {token.lower(): label for token, label in task.tokens.items()}
    truncated = choice.get("finish_reason") == "length"
    pred_text_strict = mapping.get(content.strip().lower())
    pred_text = normalize_text_answer(content, task, truncated)
    entries = ((choice.get("logprobs") or {}).get("content") or [])
    tops = entries[0].get("top_logprobs") if entries and isinstance(entries[0], dict) else None
    pred_logprob = None
    if tops:
        masses = dict.fromkeys(task.options, 0.0)
        try:
            for item in tops:
                token = str(item["token"]).strip().lower()
                if token in mapping:
                    masses[mapping[token]] += math.exp(float(item["logprob"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedResponse("bad top_logprobs entry") from exc
        if sum(masses.values()) > 0:
            pred_logprob = max(masses, key=masses.get)
    info = usage_info(payload)
    info.update(
        {
            "content": content[:40],
            "has_logprobs": bool(tops),
            "pred_logprob": pred_logprob,
            "pred_text": pred_text,
            "pred_text_strict": pred_text_strict,
        }
    )
    pred = pred_logprob if tops else pred_text
    return pred, info


def expected_label(task, label):
    if task.kind == "noul":
        return str(int(label))
    return str(label)


def d1_body(task, text):
    return {"state": text, "questions": {"answer": task.question()}}


def qwen_body(task, text, model_id):
    # Identical to harness/models.py HTTPClient.body for local chat endpoints.
    return {
        "model": model_id,
        "messages": task.messages(text),
        "temperature": 0,
        "max_tokens": QWEN_MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": QWEN_TOP_LOGPROBS,
    }


def resolve_qwen_model_id():
    try:
        payload = requests.get(base_url() + "/v1/models", timeout=10).json()
        ids = [row.get("id") for row in payload.get("data", []) if row.get("id")]
    except (requests.RequestException, ValueError, AttributeError):
        return QWEN_MODEL_ID
    if QWEN_MODEL_ID in ids or not ids:
        return QWEN_MODEL_ID
    print(f"server model id is {ids[0]!r}; chat requests will use it", flush=True)
    return ids[0]


def run_jobs(url, bodies, parallel, handle):
    """Send bodies with at most `parallel` in flight. handle(index, payload, elapsed)
    or handle(index, exc, None) runs per request, inside the worker. Returns the real
    elapsed wall-clock seconds for the whole batch (time.perf_counter around it)."""
    counter = {"done": 0}
    lock = threading.Lock()

    def one(indexed):
        index, body = indexed
        try:
            payload, elapsed = post_json(url, body)
        except Exception as exc:  # noqa: BLE001 — one bad request should not kill the pool
            handle(index, exc, None)
        else:
            handle(index, payload, elapsed)
        with lock:
            counter["done"] += 1
            done = counter["done"]
        if done % 50 == 0:
            print(f"  {done}/{len(bodies)}", flush=True)

    started = time.perf_counter()
    if parallel == 1:
        for indexed in enumerate(bodies):
            one(indexed)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(one, enumerate(bodies)))
    return time.perf_counter() - started


def evaluate(url, bodies, labels, parallel, parse):
    """Runs a batch and classifies every request as http_error, malformed,
    unparsed (well-formed but no valid answer) or parsed. Parsing happens inside
    the try, so any parser failure counts as malformed."""
    records = [None] * len(bodies)

    def handle(index, payload, elapsed):
        if elapsed is None:
            records[index] = {"status": "http_error", "error": str(payload)[:300]}
            return
        record = {"latency_ms": elapsed, "label": labels[index]}
        try:
            pred, info = parse(payload)
        except Exception as exc:  # noqa: BLE001
            record.update(status="malformed", error=f"{type(exc).__name__}: {exc}"[:300])
        else:
            record.update(info)
            record["pred"] = pred
            record["status"] = "parsed" if pred is not None else "unparsed"
        records[index] = record

    wall = run_jobs(url, bodies, parallel, handle)
    return records, wall


def mean_of(records, key):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return (sum(values) / len(values)) if values else None


def summarize(records, wall_s, k=1, extra=None):
    n = len(records)
    ok = [r for r in records if r["status"] != "http_error"]
    latencies = [r["latency_ms"] for r in ok]
    count = {s: sum(1 for r in records if r["status"] == s) for s in ("http_error", "malformed", "unparsed", "parsed")}
    labelled = [r for r in records if r["status"] == "parsed" and r.get("label") is not None]
    correct = sum(1 for r in labelled if r["pred"] == r["label"])
    row = {
        "n": n,
        "n_http_ok": len(ok),
        "n_http_errors": count["http_error"],
        "n_malformed": count["malformed"],
        "n_unparsed": count["unparsed"],
        "n_parsed": count["parsed"],
        "n_correct": correct,
        "accuracy_on_parsed": (correct / count["parsed"]) if count["parsed"] else None,
        "end_to_end_accuracy": (correct / n) if n else None,
        "parse_rate": (count["parsed"] / n) if n else None,
        "error_rate": (count["http_error"] / n) if n else None,
        "malformed_rate": (count["malformed"] / n) if n else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_min_ms": min(latencies) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "latency_mean_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "wall_s": wall_s,
        "k": k,
        "throughput_requests_per_s": (len(ok) / wall_s) if wall_s > 0 else None,
        "throughput_decisions_per_s": (len(ok) * k / wall_s) if wall_s > 0 else None,
        "mean_prompt_tokens": mean_of(ok, "prompt_tokens"),
        "mean_prompt_ms": mean_of(ok, "prompt_ms"),
        "mean_predicted_n": mean_of(ok, "predicted_n"),
        "mean_predicted_ms": mean_of(ok, "predicted_ms"),
    }
    if any("pred_text" in r for r in ok):
        text_parsed = [r for r in ok if r.get("pred_text") is not None]
        strict = [r for r in ok if r.get("pred_text_strict") is not None]
        both = [r for r in ok if r.get("pred_text") is not None and r.get("pred_logprob") is not None]
        row.update(
            {
                "qwen_n_with_logprobs": sum(1 for r in ok if r.get("has_logprobs")),
                "qwen_text_parse_rate_strict": (len(strict) / n) if n else None,
                "qwen_text_parse_rate_normalized": (len(text_parsed) / n) if n else None,
                "qwen_text_end_to_end_accuracy": (
                    sum(1 for r in text_parsed if r["pred_text"] == r.get("label")) / n
                ) if n else None,
                "qwen_text_logprob_agreement": (
                    sum(1 for r in both if r["pred_text"] == r["pred_logprob"]) / len(both)
                ) if both else None,
            }
        )
    row.update(extra or {})
    errors = [r["error"] for r in records if r.get("error")]
    if errors:
        row["error_examples"] = errors[:3]
        print(f"  http_errors {count['http_error']} malformed {count['malformed']} e.g. {errors[0]}", flush=True)
    if not ok:
        raise RuntimeError(f"every request failed: {row}")
    return row


def task_batch(kind, task, items, model_id=None):
    labels = [expected_label(task, row["label"]) for row in items]
    if kind == "d1":
        url = base_url() + "/v1/systemone"
        bodies = [d1_body(task, row["text"]) for row in items]

        def parse(payload):
            return parse_d1(payload, task)
    else:
        url = base_url() + "/v1/chat/completions"
        bodies = [qwen_body(task, row["text"], model_id) for row in items]

        def parse(payload):
            return parse_qwen(payload, task)
    return url, bodies, labels, parse


def warmup(kind, items, parallel, model_id=None):
    """WARMUP_REQUESTS unmeasured requests at the run's concurrency."""
    task = TASKS["spam"]
    rows = (items * WARMUP_REQUESTS)[:WARMUP_REQUESTS]
    url, bodies, labels, parse = task_batch(kind, task, rows, model_id)
    records, wall = evaluate(url, bodies, labels, parallel, parse)
    ok = sum(1 for r in records if r["status"] != "http_error")
    print(f"warmup {ok}/{len(records)} ok in {wall:.2f}s", flush=True)
    if ok == 0:
        raise RuntimeError(f"warmup failed: {records[0].get('error')}")


def bench_task(kind, task, items, parallel, rep, model_id=None):
    url, bodies, labels, parse = task_batch(kind, task, items, model_id)
    print(f"rep {rep} {kind} {task.name} concurrency={parallel} n={len(bodies)}", flush=True)
    records, wall = evaluate(url, bodies, labels, parallel, parse)
    row = summarize(
        records, wall,
        extra={"model": KIND_TO_MODEL[kind], "task": task.name, "parallel": parallel, "rep": rep},
    )
    return row, [r.get("latency_ms") for r in records if r["status"] != "http_error"]


def multiq_items(routing, n):
    return random.Random(SEED).sample(routing, min(n, len(routing)))


def multiq_k_order():
    ks = list(MULTIQ_KS)
    random.Random(SEED).shuffle(ks)
    return ks


def run_multiq(items):
    """d1 only, concurrency 1. Each k is one batch over the same items, timed with
    time.perf_counter around the whole batch, so wall_s is real elapsed time."""
    url = base_url() + "/v1/systemone"
    team = TASKS["routing"]
    rows = []
    for k in multiq_k_order():
        names = list(QUESTIONS)[:k]
        questions = {name: QUESTIONS[name] for name in names}
        bodies = [{"state": item["text"], "questions": questions} for item in items]
        labels = [expected_label(team, item["label"]) for item in items]

        def parse(payload, names=names):
            answers = d1_answers(payload)
            missing = [name for name in names if name not in answers]
            if missing:
                raise MalformedResponse(f"missing answers {missing}")
            return parse_d1(payload, team, "team")

        print(f"d1 multiq k={k} n={len(bodies)}", flush=True)
        records, wall = evaluate(url, bodies, labels, 1, parse)
        row = summarize(records, wall, k=k, extra={"model": "d1-3B", "task": "routing-multiq", "parallel": 1})
        row["team_accuracy_note"] = "accuracy fields score the 'team' (routing) answer only"
        rows.append(row)
    return sorted(rows, key=lambda row: row["k"])


def pool_rows(rows, latencies):
    """Pooled stats per (model, task, concurrency) across repetitions. Latency
    percentiles pool every request; throughput is total successes / total wall.
    Accuracy is taken from rep 1 (the full item set) because rep 2 re-scores a subset."""
    pooled = []
    keys = []
    for row in rows:
        key = (row["model"], row["task"], row["parallel"])
        if key not in keys:
            keys.append(key)
    for key in keys:
        group = [row for row in rows if (row["model"], row["task"], row["parallel"]) == key]
        lat = [value for row in group for value in latencies[(row["rep"],) + key]]
        wall = sum(row["wall_s"] for row in group)
        ok = sum(row["n_http_ok"] for row in group)
        first = next((row for row in group if row["rep"] == 1), group[0])
        pooled.append(
            {
                "model": key[0],
                "task": key[1],
                "parallel": key[2],
                "reps": [row["rep"] for row in group],
                "n": sum(row["n"] for row in group),
                "n_http_ok": ok,
                "n_http_errors": sum(row["n_http_errors"] for row in group),
                "n_malformed": sum(row["n_malformed"] for row in group),
                "error_rate": sum(row["n_http_errors"] for row in group) / max(1, sum(row["n"] for row in group)),
                "latency_p50_ms": percentile(lat, 50),
                "latency_p95_ms": percentile(lat, 95),
                "latency_mean_ms": (sum(lat) / len(lat)) if lat else None,
                "wall_s": wall,
                "throughput_requests_per_s": (ok / wall) if wall > 0 else None,
                "throughput_decisions_per_s": (ok / wall) if wall > 0 else None,
                "accuracy_source": f"rep {first['rep']}",
                "end_to_end_accuracy": first["end_to_end_accuracy"],
                "accuracy_on_parsed": first["accuracy_on_parsed"],
                "parse_rate": first["parse_rate"],
                "mean_prompt_tokens": first["mean_prompt_tokens"],
            }
        )
    return pooled


def fmt_ms(value):
    return "    n/a" if value is None else f"{value:7.1f}"


def fmt_rate(value):
    return "    n/a" if value is None else f"{value:7.2f}"


def fmt_frac(value):
    return "   n/a" if value is None else f"{value:6.3f}"


def print_summary(rows, pooled, multiq_rows):
    header = (
        f"{'model':<20} {'task':<8} {'rep':>4} {'conc':>4} {'n':>4} {'p50 ms':>7} {'p95 ms':>7} "
        f"{'req/s':>7} {'acc_e2e':>7} {'acc_prs':>7} {'parse':>6} {'err':>6} {'ptok':>6}"
    )
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows + pooled:
        rep = "pool" if "reps" in row else str(row["rep"])
        ptok = row.get("mean_prompt_tokens")
        print(
            f"{row['model']:<20} {row['task']:<8} {rep:>4} {row['parallel']:>4} {row['n']:>4} "
            f"{fmt_ms(row['latency_p50_ms'])} {fmt_ms(row['latency_p95_ms'])} "
            f"{fmt_rate(row['throughput_requests_per_s'])} {fmt_frac(row.get('end_to_end_accuracy')):>7} "
            f"{fmt_frac(row.get('accuracy_on_parsed')):>7} {fmt_frac(row.get('parse_rate'))} "
            f"{fmt_frac(row.get('error_rate'))} {'   n/a' if ptok is None else f'{ptok:6.0f}'}",
            flush=True,
        )
    print("\nmultiq (d1, concurrency 1, -np 1, rep 1)", flush=True)
    print(f"{'k':>3} {'n':>4} {'p50 ms':>7} {'p95 ms':>7} {'mean ms':>7} {'req/s':>7} {'dec/s':>7} {'team acc':>8}", flush=True)
    for row in multiq_rows:
        print(
            f"{row['k']:>3} {row['n']:>4} {fmt_ms(row['latency_p50_ms'])} {fmt_ms(row['latency_p95_ms'])} "
            f"{fmt_ms(row['latency_mean_ms'])} {fmt_rate(row['throughput_requests_per_s'])} "
            f"{fmt_rate(row['throughput_decisions_per_s'])} {fmt_frac(row['end_to_end_accuracy']):>8}",
            flush=True,
        )


def round_row(row):
    out = {}
    for key, value in row.items():
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


def download_models():
    models = workspace()["models"]
    paths = {}
    for key, spec in MODELS.items():
        dest = models / spec["file"]
        download(spec["url"], dest, min_bytes=100_000_000)
        if dest.stat().st_size != spec["bytes"]:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{dest} is {dest.stat().st_size} bytes, expected {spec['bytes']}")
        with open(dest, "rb") as handle:
            magic = handle.read(4)
        if magic != b"GGUF":
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{dest} is not a GGUF file")
        marker = dest.with_name(dest.name + ".sha256-verified")
        stamp = f"{spec['sha256']} {dest.stat().st_size} {int(dest.stat().st_mtime)}"
        if marker.exists() and marker.read_text().strip() == stamp:
            print(f"{spec['file']} sha256 already verified", flush=True)
        else:
            print(f"hashing {spec['file']} ...", flush=True)
            digest = sha256_file(dest)
            if digest != spec["sha256"]:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"sha256 mismatch for {spec['file']}: got {digest}, expected {spec['sha256']}")
            marker.write_text(stamp + "\n")
            print(f"{spec['file']} sha256 OK {digest[:12]}", flush=True)
        paths[key] = dest
    return paths


def run_model(binary, env, kind, model_path, items, rep, rows, latencies, servers, multiq_state, started):
    """One model in one repetition: a -np 1 server for concurrency 1, then a
    restarted -np 4 server for concurrency 4."""
    model = KIND_TO_MODEL[kind]
    for parallel in CONCURRENCIES:
        server = start_server(binary, env, model_path, MODELS[model]["alias"], parallel)
        try:
            model_id = resolve_qwen_model_id() if kind == "qwen" else None
            warmup(kind, items["spam"], parallel, model_id)
            for task_name in ("spam", "routing"):
                row, lat = bench_task(kind, TASKS[task_name], items[task_name], parallel, rep, model_id)
                rows.append(row)
                latencies[(rep, model, task_name, parallel)] = lat
            if kind == "d1" and rep == 1 and parallel == 1:
                elapsed = time.perf_counter() - started
                if elapsed > MULTIQ_CUTOFF_S:
                    n_multi = MULTIQ_SHORT
                    reason = (
                        f"{n_multi} routing items (seeded sample, seed {SEED}) because {elapsed / 60:.1f} min "
                        f"had elapsed (cutoff {MULTIQ_CUTOFF_S // 60} min)."
                    )
                else:
                    n_multi = MULTIQ_FULL
                    reason = f"{n_multi} routing items, seeded random sample (random.Random({SEED}).sample)."
                print("multiq:", reason, flush=True)
                sample = multiq_items(items["routing_all"], n_multi)
                multiq_state.update(
                    n_items=len(sample), reason=reason, item_ids=[item["id"] for item in sample],
                    rows=run_multiq(sample),
                )
            record = server_record(server)
            record["gpu_mem_used_after_runs_mib"] = gpu_mem_mib()
            record.update(model=model, rep=rep)
            servers.append(record)
        finally:
            close_server(server)


def main():
    started = time.perf_counter()
    cuda, gpu = gpu_query()
    binary, env = ensure_llama_server(cuda)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env)
    version_text = ((version.stdout or "") + (version.stderr or "")).strip().splitlines()
    print(version_text[0] if version_text else "llama-server", flush=True)
    models = download_models()
    spam, routing, dataset_meta = extract_eval_zip(locate_zip(), workspace()["root"])

    rep_plan = [
        {"rep": 1, "order": ["d1", "qwen"], "items": {"spam": spam, "routing": routing}},
    ]
    rows, latencies, servers, multiq_state = [], {}, [], {}
    # Rep 1: full sets, d1 first.
    for kind in rep_plan[0]["order"]:
        items = dict(rep_plan[0]["items"], routing_all=routing)
        run_model(binary, env, kind, models[KIND_TO_MODEL[kind]], items, 1, rows, latencies, servers, multiq_state, started)

    # Rep 2: reversed order on a seeded subset, sized to the remaining budget.
    elapsed = time.perf_counter() - started
    if elapsed > REP2_SKIP_AFTER_S:
        rep2_note = f"rep 2 skipped: {elapsed / 60:.1f} min elapsed (limit {REP2_SKIP_AFTER_S // 60} min)."
        n_rep2 = 0
    else:
        n_rep2 = REP2_ITEMS_SHORT if elapsed > REP2_SHRINK_AFTER_S else REP2_ITEMS
        rep2_note = (
            f"rep 2: Qwen then d1 on {n_rep2} items per task, random.Random({SEED + 2}).sample of each "
            f"rep 1 set ({elapsed / 60:.1f} min elapsed before rep 2)."
        )
    print(rep2_note, flush=True)
    if n_rep2:
        rng = random.Random(SEED + 2)
        subset = {
            "spam": rng.sample(spam, min(n_rep2, len(spam))),
            "routing": rng.sample(routing, min(n_rep2, len(routing))),
        }
        rep_plan.append({"rep": 2, "order": ["qwen", "d1"], "items": subset})
        for kind in rep_plan[1]["order"]:
            items = dict(subset, routing_all=routing)
            run_model(binary, env, kind, models[KIND_TO_MODEL[kind]], items, 2, rows, latencies, servers, multiq_state, started)

    pooled = pool_rows(rows, latencies)
    multiq_rows = multiq_state.get("rows", [])
    print_summary(rows, pooled, multiq_rows)

    model_config = {}
    for model, spec in MODELS.items():
        mine = [s for s in servers if s["model"] == model]
        model_config[model] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "url": spec["url"],
            "file": spec["file"],
            "sha256": spec["sha256"],
            "sha256_verified": True,
            "bytes": models[model].stat().st_size,
            "endpoint": "/v1/systemone" if model == "d1-3B" else "/v1/chat/completions",
            "flash_attn": sorted({s["flash_attn"] for s in mine}),
            "layers_offloaded": sorted({f"{s['offloaded_layers']}/{s['total_layers']}" for s in mine}),
            "gpu_mem_used_after_load_mib": {f"rep{s['rep']}-np{s['n_slots']}": s["gpu_mem_used_after_load_mib"] for s in mine},
            "gpu_mem_delta_mib": {f"rep{s['rep']}-np{s['n_slots']}": s["gpu_mem_delta_mib"] for s in mine},
            "gpu_mem_used_after_runs_mib": {f"rep{s['rep']}-np{s['n_slots']}": s["gpu_mem_used_after_runs_mib"] for s in mine},
        }
    report = {
        "created_unix": time.time(),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "gpu": gpu,
        "scope": (
            "End-to-end application performance as served by llama-server (HTTP latency, throughput, "
            "accuracy). Not a measure of architecture efficiency: the two endpoints build different "
            "prompts and do different server-side work; see mean_prompt_tokens and timings."
        ),
        "llama_server": {
            "binary": str(binary),
            "version": version_text[:3],
            "n_gpu_layers": "all (full offload verified from the load log on every start)",
            "ctx_per_slot": CTX_PER_SLOT,
            "concurrency_to_slots": {str(c): c for c in CONCURRENCIES},
            "prompt_cache": False,
            "servers": servers,
        },
        "models": model_config,
        "dataset": dataset_meta,
        "settings": {
            "qwen_body": {
                "temperature": 0,
                "max_tokens": QWEN_MAX_TOKENS,
                "logprobs": True,
                "top_logprobs": QWEN_TOP_LOGPROBS,
            },
            "qwen_prediction": (
                "harness/models.py rule: argmax of first-token top_logprobs mass over valid answer tokens; "
                "normalized text parse used only when logprobs are absent, and reported separately"
            ),
            "constrained_decoding": "none (no grammar or logit_bias), to keep the CPU harness request unchanged",
            "warmup": f"{WARMUP_REQUESTS} requests per server start at the run's concurrency, not measured",
            "repetitions": [
                {"rep": plan["rep"], "order": plan["order"], "n_items": {t: len(plan["items"][t]) for t in ("spam", "routing")}}
                for plan in rep_plan
            ],
            "seed": SEED,
            "percentile": "nearest-rank",
            "throughput": "successful HTTP requests / real elapsed wall-clock of the batch (time.perf_counter)",
            "decisions_per_s": "throughput_requests_per_s * k (k = questions per request; 1 for spam/routing)",
            "end_to_end_accuracy": "correct / all requests (http errors, malformed and unparsed count as wrong)",
            "accuracy_on_parsed": "correct / parsed",
        },
        "notes": [
            "Prompts and question objects are copied from harness/tasks.py and harness/multiq.py.",
            "Routing criteria use SUBSCRIPTION, not NEWSLETTER.",
            "Concurrency 1 runs on a server started with -np 1; concurrency 4 on a fresh server with -np 4.",
            "The multi-question test uses a seeded random sample of routing items (the CPU harness uses the first 30).",
            "multiq runs each k as its own timed batch (k order shuffled with the same seed).",
            rep2_note,
            multiq_state.get("reason", ""),
        ],
        "results": [round_row(row) for row in rows],
        "pooled": [round_row(row) for row in pooled],
        "multiq": {
            "n_items": multiq_state.get("n_items"),
            "reason": multiq_state.get("reason"),
            "item_ids": multiq_state.get("item_ids"),
            "questions": list(QUESTIONS),
            "k_order": multiq_k_order(),
            "parallel": 1,
            "rows": [round_row(row) for row in multiq_rows],
        },
    }
    out = workspace()["results"]
    full = dict(report)
    full["raw_latencies_ms"] = {
        f"rep{key[0]}|{key[1]}|{key[2]}|c{key[3]}": [round(v, 2) for v in values] for key, values in latencies.items()
    }
    out.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out} (includes raw per-request latencies)", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
