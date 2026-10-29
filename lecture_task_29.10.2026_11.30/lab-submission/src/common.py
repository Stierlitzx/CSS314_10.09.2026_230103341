"""Paths, serialization and native-kernel loading shared by the experiments."""
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVIDENCE = ROOT.parent / "evidence"
DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def params():
    return read_json(RESULTS / "salted_params.json")


def pointer(array):
    assert array.dtype == np.float64 and array.flags.c_contiguous
    return array.ctypes.data_as(DOUBLE_PTR)


def digest(array, student_id):
    """Canonical little-endian float64, rounded to six decimals, salted by ID."""
    data = np.round(np.asarray(array, dtype=np.float64), 6).astype("<f8")
    return hashlib.sha256(student_id.encode() + b"\0" + data.tobytes(order="C")).hexdigest()


def build_native():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    target = build / ("kernels.dll" if os.name == "nt" else "kernels.so")
    source = ROOT / "src" / "kernels.c"
    candidates = [os.environ.get("CC"), shutil.which("gcc"), shutil.which("clang"), shutil.which("cc")]
    if os.name == "nt":
        # An old 32-bit MinGW can precede a modern 64-bit toolchain on PATH.
        candidates += [str(p) for p in (Path(os.environ["LOCALAPPDATA"]) / "Programs").glob("*/bin/mingw/bin/gcc.exe")]
    compiler = None
    for candidate in candidates:
        if not candidate:
            continue
        machine = subprocess.check_output([candidate, "-dumpmachine"], text=True).strip()
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8 or "x86_64" in machine:
            compiler = candidate
            break
    if not compiler:
        raise RuntimeError("GCC or Clang is required for the ctypes kernels")
    command = [compiler, "-O2", "-std=c99", "-shared", "-static-libgcc", "-fno-fast-math",
               "-ffp-contract=off", "-fno-tree-vectorize"]
    if os.name != "nt":
        command.append("-fPIC")
    command += [str(source), "-o", str(target)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    write_json(EVIDENCE / "native_build.json", {
        "command": command, "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "note": "Scalar kernels; no BLAS, fast-math, reassociation, or automatic vectorization."})
    return target


def native():
    target = ROOT / "build" / ("kernels.dll" if os.name == "nt" else "kernels.so")
    if not target.exists():
        build_native()
    lib = ctypes.CDLL(str(target))
    lib.stride_sum.argtypes = [DOUBLE_PTR, ctypes.c_size_t, ctypes.c_size_t]
    lib.stride_sum.restype = ctypes.c_double
    for name in ("matmul_ijk", "matmul_ikj"):
        getattr(lib, name).argtypes = [DOUBLE_PTR] * 3 + [ctypes.c_int, ctypes.c_int]
        getattr(lib, name).restype = None
    lib.stencil.argtypes = [DOUBLE_PTR, DOUBLE_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.stencil.restype = None
    lib.counter_loop.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int64]
    lib.counter_loop.restype = None
    return lib


def cpu_order():
    host = read_json(RESULTS / "host_profile.json")
    groups = host["cpu"]["core_logical_cpu_ids"]
    return [g[0] for g in groups] + [c for g in groups for c in g[1:]]


def pin_cpu(cpu):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_workers(target, arguments):
    """Spawn, wait for readiness, release together, then collect completion.

    Primary wall time includes start-event/completion-queue overhead, but excludes
    spawn/import/setup and join. Full lifecycle is reported separately.
    """
    context = mp.get_context("spawn")
    queue, gate = context.Queue(), context.Event()
    processes = [context.Process(target=target, args=(*args, gate, queue)) for args in arguments]
    lifecycle_start = time.perf_counter_ns()
    try:
        for process in processes:
            process.start()
        for _ in processes:
            message = queue.get(timeout=180)
            if message[0] != "ready":
                raise RuntimeError(f"Worker setup failed: {message}")
        start = time.perf_counter_ns()
        gate.set()
        completed = [queue.get(timeout=180) for _ in processes]
        elapsed = time.perf_counter_ns() - start
        if any(message[0] != "done" for message in completed):
            raise RuntimeError(f"Worker failed: {completed}")
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f"Worker exit code {process.exitcode}")
        return elapsed, time.perf_counter_ns() - lifecycle_start, completed
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()
        queue.join_thread()
