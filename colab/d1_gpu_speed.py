#!/usr/bin/env python3
"""Colab T4 speed test: Liquid d1-3B vs Qwen2.5-3B-Instruct, both Q4_K_M.

Same spam and routing prompts as harness/tasks.py (the routing label is
SUBSCRIPTION, not NEWSLETTER) and the same multi-question set as
harness/multiq.py. Measures HTTP latency and throughput at parallelism
1 and 4.

Inference is llama.cpp llama-server b11541 with CUDA, all layers on the
GPU (-ngl all) and 4 slots (-np 4). d1 is POST /v1/systemone. Qwen is
POST /v1/chat/completions at temperature 0 and max_tokens 4, without
logprobs (this run is latency-focused). Prompt cache is off so each
call pays a full prefill.

The notebook colab/d1_gpu_speed.ipynb embeds this file as its main cell.
Its last cell disconnects the Colab runtime. This script does not.

Upload colab/d1data.zip to /content/d1data.zip before running on Colab.
"""

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

D1_URL = "https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/d1-3B-Q4_K_M.gguf"
QWEN_URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
D1_NAME = "d1-3B-Q4_K_M.gguf"
QWEN_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN_MODEL_ID = "Qwen2.5-3B-Instruct"

HOST = "127.0.0.1"
PORT = 8081
N_PARALLEL_SLOTS = 4
CTX_SIZE = 8192  # total KV pool; llama-server divides it across the 4 slots
MIN_GPU_MIB = 400
MULTIQ_FULL = 30
MULTIQ_SHORT = 10
MULTIQ_CUTOFF_S = 15 * 60

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

WHEEL_TAGS = (
    (13, 2, "cu132"),
    (13, 0, "cu130"),
    (12, 5, "cu125"),
    (12, 4, "cu124"),
    (12, 3, "cu123"),
    (12, 2, "cu122"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
)
CMAKE_FALLBACK = (
    'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" '
    "pip install llama-cpp-python --no-binary llama-cpp-python"
)


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


def wheel_tag(cuda):
    for major, minor, tag in WHEEL_TAGS:
        if cuda >= (major, minor):
            return tag
    return "cu124"


def binding_has_cuda():
    code = (
        "import llama_cpp\n"
        "print('yes' if llama_cpp.llama_supports_gpu_offload() else 'no')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip().endswith("yes")


def pip_install(args):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    result = run(cmd, capture_output=True, text=True)
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 and "externally-managed-environment" in combined:
        result = run(cmd + ["--break-system-packages"], capture_output=True, text=True)
        combined = (result.stdout or "") + (result.stderr or "")
    return result.returncode, combined[-2000:]


def install_python_wheel(cuda):
    """Best-effort prebuilt CUDA wheel. The timed runs use llama-server, which
    is the binary that exposes /v1/systemone. --only-binary avoids a source
    build that would blow the 20 minute budget."""
    if binding_has_cuda():
        print("llama-cpp-python CUDA wheel already importable; not reinstalling", flush=True)
        return {"installed": True, "detail": "already importable with GPU offload"}
    tag = wheel_tag(cuda)
    # Primary index is the CUDA wheel index so pip does not prefer the PyPI CPU wheel.
    # PyPI stays as an extra index for the package's dependencies.
    url = f"https://abetlen.github.io/llama-cpp-python/whl/{tag}"
    code, tail = pip_install(
        [
            "-q", "llama-cpp-python", "--only-binary=llama-cpp-python",
            "--index-url", url, "--extra-index-url", "https://pypi.org/simple",
        ]
    )
    if code != 0 or not binding_has_cuda():
        print("llama-cpp-python CUDA wheel was not installed.", flush=True)
        print("The speed test still uses the llama-server CUDA binary (systemone + -np 4).", flush=True)
        print("Source-build fallback, only if you need the Python binding (often >10 min):", flush=True)
        print("  " + CMAKE_FALLBACK, flush=True)
        if tail:
            print(tail[-800:], flush=True)
        return {"installed": False, "wheel": tag, "detail": "no usable prebuilt wheel", "fallback": CMAKE_FALLBACK}
    print(f"installed llama-cpp-python CUDA wheel ({tag}); benchmark uses llama-server", flush=True)
    return {"installed": True, "wheel": tag, "detail": "prebuilt wheel"}


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
    for want in ("spam.jsonl", "routing.jsonl"):
        data = read_jsonl(dest / want)
        if not data or not {"id", "text", "label"} <= set(data[0]):
            raise RuntimeError(f"{want} is empty or missing id/text/label")
        rows[want] = data
        print(f"{want}: {len(data)} items", flush=True)
    return rows["spam.jsonl"], rows["routing.jsonl"]


def server_command(binary, model_path, alias, flash):
    threads = str(min(8, os.cpu_count() or 2))
    return [
        str(binary),
        "-m", str(model_path),
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", "all",
        "-np", str(N_PARALLEL_SLOTS),
        "-c", str(CTX_SIZE),
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


def log_text(path: Path):
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-12000:]


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


def start_server(binary, env, model_path, alias):
    paths = workspace()
    log_path = paths["log"]
    last_error = None
    for flash in ("on", "off"):
        if log_path.exists():
            log_path.unlink()
        cmd = server_command(binary, model_path, alias, flash)
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
            text = wait_until_loaded(proc, log_path)
        except Exception as exc:
            last_error = exc
            stop_server(proc)
            log_handle.close()
            print(f"start with -fa {flash} failed: {exc}", flush=True)
            continue
        mem_after = gpu_mem_mib()
        delta = mem_after - mem_before
        offload = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", text)
        print(f"server ready (-fa {flash}), GPU memory +{delta} MiB", flush=True)
        if offload:
            print(f"offloaded {offload.group(1)}/{offload.group(2)} layers to GPU", flush=True)
        if delta < MIN_GPU_MIB and not (offload and int(offload.group(1)) > 0):
            stop_server(proc)
            log_handle.close()
            last_error = RuntimeError(
                f"GPU memory rose only {delta} MiB; the model is not fully on the GPU.\n" + text[-1500:]
            )
            continue
        return {
            "proc": proc,
            "log": log_handle,
            "cmd": cmd,
            "gpu_mem_delta_mib": delta,
            "flash": flash,
        }
    raise RuntimeError(f"could not start llama-server for {model_path}") from last_error


def base_url():
    return f"http://{HOST}:{PORT}"


_thread_local = threading.local()


def http_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def post_json(url, body):
    started = time.perf_counter()
    try:
        response = http_session().post(url, json=body, timeout=(10, 60))
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text[:500]}
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {str(payload)[:400]}")
    return payload, elapsed


def parse_d1(payload, task, name="answer"):
    answers = payload.get("answers", payload.get("results", payload))
    if not isinstance(answers, dict) or name not in answers:
        return None
    answer = answers[name]
    if task.kind == "noul":
        if "noul" not in answer:
            return None
        p_yes = float(answer["noul"])
        values = {"0": 1 - p_yes, "1": p_yes}
        return max(values, key=values.get)
    probabilities = answer.get("probabilities") or {}
    if probabilities:
        return max(probabilities, key=lambda key: float(probabilities[key]))
    return answer.get("choice")


def parse_qwen(payload, task):
    try:
        content = payload["choices"][0].get("message", {}).get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None
    mapping = {token.lower(): label for token, label in task.tokens.items()}
    return mapping.get(content.strip().lower())


def expected_label(task, label):
    if task.kind == "noul":
        return str(int(label))
    return str(label)


def d1_body(task, text):
    return {"state": text, "questions": {"answer": task.question()}}


def qwen_body(task, text, model_id):
    return {
        "model": model_id,
        "messages": task.messages(text),
        "temperature": 0,
        "max_tokens": 4,
    }


def resolve_qwen_model_id():
    try:
        payload = requests.get(base_url() + "/v1/models", timeout=10).json()
        ids = [row.get("id") for row in payload.get("data", []) if row.get("id")]
    except (requests.RequestException, ValueError):
        return QWEN_MODEL_ID
    if QWEN_MODEL_ID in ids or not ids:
        return QWEN_MODEL_ID
    print(f"server model id is {ids[0]!r}; chat requests will use it", flush=True)
    return ids[0]


def warmup(url, body):
    last = None
    for _ in range(3):
        try:
            payload, elapsed = post_json(url, body)
            print(f"warmup {elapsed:.1f} ms", flush=True)
            return payload, elapsed
        except RuntimeError as exc:
            last = exc
            time.sleep(1)
    raise RuntimeError(f"warmup failed: {last}")


def summarize_latencies(latencies, errors, n_parsed, n_correct, wall_s, extra):
    n_ok = len(latencies)
    n = n_ok + errors
    mean = sum(latencies) / n_ok if n_ok else None
    row = {
        "n": n,
        "n_ok": n_ok,
        "n_error": errors,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_parsed) if n_parsed else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_min_ms": min(latencies) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "latency_mean_ms": mean,
        "throughput_decisions_per_s": (n_ok / wall_s) if wall_s > 0 else None,
        "wall_s": wall_s,
    }
    row.update(extra)
    if n_ok == 0 or (errors > n_ok and errors > 10):
        raise RuntimeError(f"too many failed requests: {row}")
    return row


def run_parallel(url, jobs, parallel, parse):
    """jobs are (body, label). Latency is the HTTP time once a worker sends.
    Throughput uses wall clock with at most `parallel` requests in flight.
    """
    state = {"latencies": [], "errors": 0, "samples": [], "parsed": 0, "correct": 0, "done": 0}
    lock = threading.Lock()

    def one(job):
        body, label = job
        try:
            payload, elapsed = post_json(url, body)
        except Exception as exc:  # noqa: BLE001 — one bad response should not kill the pool
            with lock:
                state["errors"] += 1
                state["done"] += 1
                if len(state["samples"]) < 5:
                    state["samples"].append(str(exc)[:300])
                done = state["done"]
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
            return
        pred = parse(payload)
        with lock:
            state["latencies"].append(elapsed)
            if pred is not None:
                state["parsed"] += 1
                if label is not None and pred == label:
                    state["correct"] += 1
            state["done"] += 1
            done = state["done"]
        if done % 50 == 0:
            print(f"  {done}/{len(jobs)}", flush=True)

    started = time.perf_counter()
    if parallel == 1:
        for job in jobs:
            one(job)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(one, jobs))
    wall = time.perf_counter() - started
    if state["samples"]:
        print(f"  errors {state['errors']} e.g. {state['samples'][0]}", flush=True)
    return state["latencies"], state["errors"], state["parsed"], state["correct"], wall


def bench_task(kind, task, items, parallel, model_id=None):
    if kind == "d1":
        url = base_url() + "/v1/systemone"
        jobs = [(d1_body(task, row["text"]), expected_label(task, row["label"])) for row in items]

        def parse(payload):
            return parse_d1(payload, task)
    else:
        url = base_url() + "/v1/chat/completions"
        jobs = [
            (qwen_body(task, row["text"], model_id), expected_label(task, row["label"])) for row in items
        ]

        def parse(payload):
            return parse_qwen(payload, task)
    print(f"{kind} {task.name} parallel={parallel} n={len(jobs)}", flush=True)
    latencies, errors, n_parsed, n_correct, wall = run_parallel(url, jobs, parallel, parse)
    return summarize_latencies(
        latencies, errors, n_parsed, n_correct, wall,
        {"model": "d1-3B" if kind == "d1" else "qwen2.5-3b-instruct", "task": task.name, "parallel": parallel},
    )


def multiq_jobs(items):
    rng = random.Random(42)
    jobs = []
    for item in items:
        ks = [1, 2, 3, 5]
        rng.shuffle(ks)
        for k in ks:
            questions = dict(list(QUESTIONS.items())[:k])
            body = {"state": item["text"], "questions": questions}
            jobs.append((body, k, list(questions)))
    return jobs


def run_multiq(items):
    url = base_url() + "/v1/systemone"
    jobs = multiq_jobs(items)
    by_k = {k: [] for k in (1, 2, 3, 5)}
    errors = {k: 0 for k in (1, 2, 3, 5)}
    print(f"d1 multiq sequential n_items={len(items)} requests={len(jobs)}", flush=True)
    for index, (body, k, names) in enumerate(jobs, start=1):
        try:
            payload, elapsed = post_json(url, body)
        except Exception as exc:  # noqa: BLE001
            errors[k] += 1
            if errors[k] == 1:
                print(f"  multiq k={k} error {exc}", flush=True)
            continue
        answers = payload.get("answers", payload.get("results", payload))
        if not isinstance(answers, dict) or not set(names).issubset(answers):
            errors[k] += 1
            continue
        by_k[k].append(elapsed)
        if index % 40 == 0:
            print(f"  multiq {index}/{len(jobs)}", flush=True)
    rows = []
    for k in (1, 2, 3, 5):
        latencies = by_k[k]
        wall = sum(latencies) / 1000 if latencies else 0
        rows.append({
            "model": "d1-3B",
            "task": "routing-multiq",
            "k": k,
            "parallel": 1,
            "n": len(latencies) + errors[k],
            "n_ok": len(latencies),
            "n_error": errors[k],
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "latency_mean_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "throughput_decisions_per_s": (len(latencies) / wall) if wall else None,
            "wall_s": wall,
        })
    if sum(row["n_ok"] for row in rows) == 0:
        raise RuntimeError("multiq produced no successful responses")
    return rows


def fmt_ms(value):
    return "  n/a" if value is None else f"{value:7.1f}"


def fmt_rate(value):
    return "  n/a" if value is None else f"{value:7.2f}"


def print_summary(rows, multiq_rows):
    header = f"{'model':<22} {'task':<16} {'par':>3} {'n':>5} {'p50 ms':>8} {'p95 ms':>8} {'dec/s':>8} {'acc':>6}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        acc = row.get("accuracy")
        acc_s = "  n/a" if acc is None else f"{acc:6.3f}"
        print(
            f"{row['model']:<22} {row['task']:<16} {row['parallel']:>3} {row['n']:>5} "
            f"{fmt_ms(row['latency_p50_ms'])} {fmt_ms(row['latency_p95_ms'])} "
            f"{fmt_rate(row['throughput_decisions_per_s'])} {acc_s}",
            flush=True,
        )
    print("\nmultiq (d1, sequential, latency vs questions per request)", flush=True)
    print(f"{'k':>3} {'n':>5} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8} {'dec/s':>8}", flush=True)
    for row in multiq_rows:
        print(
            f"{row['k']:>3} {row['n']:>5} {fmt_ms(row['latency_p50_ms'])} {fmt_ms(row['latency_p95_ms'])} "
            f"{fmt_ms(row['latency_mean_ms'])} {fmt_rate(row['throughput_decisions_per_s'])}",
            flush=True,
        )


def round_row(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = round(value, 3)
        else:
            out[key] = value
    return out


def download_models():
    models = workspace()["models"]
    specs = (
        ("d1-3B", D1_NAME, D1_URL),
        ("qwen2.5-3b-instruct", QWEN_NAME, QWEN_URL),
    )
    paths = {}
    for key, name, url in specs:
        dest = models / name
        download(url, dest, min_bytes=100_000_000)
        with open(dest, "rb") as handle:
            magic = handle.read(4)
        if magic != b"GGUF":
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{dest} is not a GGUF file")
        paths[key] = dest
        print(f"{name} {dest.stat().st_size} bytes", flush=True)
    return paths


def main():
    started = time.perf_counter()
    cuda, gpu = gpu_query()
    wheel = install_python_wheel(cuda)
    binary, env = ensure_llama_server(cuda)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env)
    version_text = ((version.stdout or "") + (version.stderr or "")).strip().splitlines()
    print(version_text[0] if version_text else "llama-server", flush=True)
    models = download_models()
    spam, routing = extract_eval_zip(locate_zip(), workspace()["root"])
    sets = {"spam": spam, "routing": routing}
    rows = []
    server = None
    multiq_meta = {}
    try:
        for kind, model_key, alias, filename in (
            ("d1", "d1-3B", "d1-3B", D1_NAME),
            ("qwen", "qwen2.5-3b-instruct", QWEN_MODEL_ID, QWEN_NAME),
        ):
            stop_server(server["proc"] if server else None)
            if server and server.get("log"):
                server["log"].close()
            server = start_server(binary, env, models[model_key], alias)
            model_id = None
            if kind == "d1":
                warm_body = d1_body(TASKS["spam"], spam[0]["text"])
                warmup(base_url() + "/v1/systemone", warm_body)
            else:
                model_id = resolve_qwen_model_id()
                warm_body = qwen_body(TASKS["spam"], spam[0]["text"], model_id)
                warmup(base_url() + "/v1/chat/completions", warm_body)
            for task_name in ("spam", "routing"):
                for parallel in (1, 4):
                    rows.append(bench_task(kind, TASKS[task_name], sets[task_name], parallel, model_id))
            if kind == "d1":
                elapsed = time.perf_counter() - started
                if elapsed > MULTIQ_CUTOFF_S:
                    n_multi = MULTIQ_SHORT
                    reason = (
                        f"Used {n_multi} routing items because {elapsed / 60:.1f} min had already elapsed "
                        f"(cutoff {MULTIQ_CUTOFF_S // 60} min). Prefer 30 when the budget allows."
                    )
                else:
                    n_multi = min(MULTIQ_FULL, len(routing))
                    reason = f"Used the first {n_multi} routing items, k in {{1, 2, 3, 5}}."
                print(reason, flush=True)
                multiq_rows = run_multiq(routing[:n_multi])
                multiq_meta = {"n_items": n_multi, "reason": reason, "rows": multiq_rows}
    finally:
        if server:
            stop_server(server["proc"])
            server["log"].close()

    multiq_rows = multiq_meta.get("rows", [])
    print_summary(rows, multiq_rows)
    report = {
        "created_unix": time.time(),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "gpu": gpu,
        "llama_server": {
            "binary": str(binary),
            "version": version_text[:3],
            "n_gpu_layers": "all",
            "parallel_slots": N_PARALLEL_SLOTS,
            "ctx_size_total": CTX_SIZE,
            "ctx_per_slot_approx": CTX_SIZE // N_PARALLEL_SLOTS,
            "prompt_cache": False,
            "flash_attn": server["flash"] if server else None,
            "command": server["cmd"] if server else None,
        },
        "llama_cpp_python": wheel,
        "models": {
            "d1-3B": {"url": D1_URL, "file": D1_NAME, "bytes": models["d1-3B"].stat().st_size},
            "qwen2.5-3b-instruct": {
                "url": QWEN_URL,
                "file": QWEN_NAME,
                "bytes": models["qwen2.5-3b-instruct"].stat().st_size,
            },
        },
        "settings": {
            "temperature": 0,
            "max_tokens": 4,
            "logprobs": False,
            "warmup": "one request per model after load, not included in metrics",
            "percentile": "nearest-rank",
            "throughput": "HTTP successes / wall seconds at the stated concurrency",
        },
        "notes": [
            "Prompts and question objects are copied from harness/tasks.py and harness/multiq.py.",
            "Routing criteria use SUBSCRIPTION, not NEWSLETTER.",
            "d1 requests are {state, questions: {answer: task question}} to /v1/systemone.",
            "Qwen requests are chat completions with the harness system and user messages.",
            "Parallel 1 sends one request at a time. Parallel 4 keeps four requests in flight on a server started with -np 4.",
            "The server stays loaded between parallel 1 and 4. Prompt cache and slot similarity are off.",
            "-c 8192 with -np 4 gives about 2048 tokens per slot, which covers these short prompts.",
            multiq_meta.get("reason", ""),
        ],
        "results": [round_row(row) for row in rows],
        "multiq": {
            "n_items": multiq_meta.get("n_items"),
            "reason": multiq_meta.get("reason"),
            "questions": list(QUESTIONS),
            "k": [1, 2, 3, 5],
            "k_order": "shuffled per item with random.Random(42), matching harness/multiq.py",
            "parallel": 1,
            "rows": [round_row(row) for row in multiq_rows],
        },
    }
    out = workspace()["results"]
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
