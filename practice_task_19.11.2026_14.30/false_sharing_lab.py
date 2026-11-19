import os
import sys
import json
import time
import ctypes
import hashlib
import platform
import statistics
import threading
import subprocess
import numpy as np
import re
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

try:
    import psutil  # used only for context-switch counts where `resource` does not exist (Windows)
except ImportError:
    psutil = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "Stierlitzx"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

# Additional lab constants (recorded in the manifest)
STRIDES = (1, 4, 8, 16)
TRIALS = 5                      # each layout is timed TRIALS times; the median is reported
BUFFER_ALIGN = 128              # 128 B alignment => 64 B aligned and on a 128 B prefetch-pair boundary
GIL_ITERATIONS = 300_000        # pure-Python bytecode baseline for Q5 (per thread)
MANIFEST_PATH = "cache_lab_manifest.json"

# Compile-time uint64 constants for the kernels
_GAMMA, _S58, _ONE = np.uint64(0x9E3779B97F4A7C15), np.uint64(58), np.uint64(1)


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    if sys.platform == "win32":
        win = _windows_topology()
        if win and win["l1d_line_size"]:
            return win["l1d_line_size"]
    return 64


# ------------------------------------------------------------------------------
# Topology helpers (physical-core grouping for affinity pinning)
# ------------------------------------------------------------------------------
class _CacheDescriptor(ctypes.Structure):
    _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                ("LineSize", ctypes.c_ushort), ("Size", ctypes.c_uint32), ("Type", ctypes.c_uint32)]


class _SlpiUnion(ctypes.Union):
    _fields_ = [("Flags", ctypes.c_ubyte), ("NodeNumber", ctypes.c_uint32),
                ("Cache", _CacheDescriptor), ("Reserved", ctypes.c_ulonglong * 2)]


class _Slpi(ctypes.Structure):
    _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_uint32), ("u", _SlpiUnion)]


def _windows_topology():
    """GetLogicalProcessorInformation: physical-core masks and the L1d line size."""
    try:
        k32 = ctypes.windll.kernel32
        size = ctypes.c_uint32(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
        n = size.value // ctypes.sizeof(_Slpi)
        buf = (_Slpi * n)()
        if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
            return None
    except Exception:
        return None
    cores, l1d = [], 0
    for e in buf:
        if e.Relationship == 0:                                  # RelationProcessorCore
            cores.append([b for b in range(64) if e.ProcessorMask >> b & 1])
        elif e.Relationship == 2 and e.u.Cache.Level == 1 and e.u.Cache.Type in (0, 2):
            l1d = int(e.u.Cache.LineSize)                        # CacheUnified / CacheData
    return {"cores": sorted(cores), "l1d_line_size": l1d}


def _linux_topology():
    cores = {}
    base = "/sys/devices/system/cpu"
    for cpu in sorted(int(d[3:]) for d in os.listdir(base) if d[3:].isdigit() and d.startswith("cpu")):
        try:
            with open(f"{base}/cpu{cpu}/topology/core_id") as f:
                core = int(f.read())
            with open(f"{base}/cpu{cpu}/topology/physical_package_id") as f:
                pkg = int(f.read())
        except OSError:
            continue
        cores.setdefault((pkg, core), []).append(cpu)
    return sorted(cores.values())


def physical_core_groups():
    if sys.platform == "win32":
        win = _windows_topology()
        return win["cores"] if win else []
    if sys.platform.startswith("linux"):
        return _linux_topology()
    return []


def cpu_model_string() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                encoding="utf-8", stderr=subprocess.DEVNULL)
            return out.strip().splitlines()[0].strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def pin_current_thread(cpu):
    """Pin the calling OS thread to one logical CPU. Returns True when the OS accepted the mask."""
    if cpu is None:
        return False
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentThread.restype = ctypes.c_void_p
        k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetThreadAffinityMask.restype = ctypes.c_size_t
        return k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << cpu) != 0
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})   # pid 0 == calling thread on Linux
        return True
    return False


def splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")
    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    groups = physical_core_groups()
    # One logical CPU from each of the first NUM_THREADS physical cores (avoids SMT siblings)
    pin_cpus = [g[0] for g in groups[:NUM_THREADS]] if len(groups) >= NUM_THREADS else [None] * NUM_THREADS
    affinity_supported = sys.platform == "win32" or hasattr(os, "sched_setaffinity")

    thread_seeds = [splitmix64(instance_seed + t) | 1 for t in range(NUM_THREADS)]
    power = None
    if psutil is not None and hasattr(psutil, "sensors_battery"):
        b = psutil.sensors_battery()
        power = None if b is None else {"battery_percent": b.percent, "ac_plugged": b.power_plugged}
    return {
        "git_user": git_user.strip(),
        "cpu_sig": cpu_sig,
        "cpu_model": cpu_model_string(),
        "platform": platform.platform(),
        "logical_cores": logical_cores,
        "physical_cores": len(groups) if groups else None,
        "core_groups": groups,
        "cache_line_size": line_size,
        "instance_seed": instance_seed,
        "thread_seeds": thread_seeds,
        "affinity_supported": affinity_supported,
        "pin_cpus": pin_cpus,
        "power": power,
    }


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed, mask):
    """Read-modify-write buffer[slot] in memory every iteration.

    `j = slot + (x & mask)` depends on the loop-carried Weyl-sequence state x. `mask` is 0 at run
    time, so j == slot always, but the compiler cannot prove that, so it cannot keep buffer[slot]
    in a register across iterations: every iteration issues a real load and a real store.
    The Weyl step is a single 1-cycle add, so the memory read-modify-write is the critical path.
    """
    x = seed
    for _ in range(iters):
        x += _GAMMA
        j = slot + (x & mask)
        buffer[j] = buffer[j] + (x >> _S58) + _ONE
    return x


@njit(nogil=True)
def naive_promotable_accumulate(buffer, slot, iters, seed):
    """Control for Q4: the textbook loop, whose fixed address LLVM is free to promote to a register."""
    x = seed
    for _ in range(iters):
        x += _GAMMA
        buffer[slot] = buffer[slot] + (x >> _S58) + _ONE
    return x


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Same arithmetic, accumulated in a local register; exactly one shared store at the end."""
    x = seed
    acc = np.uint64(0)
    for _ in range(iters):
        x += _GAMMA
        acc += (x >> _S58) + _ONE
    buffer[slot] = acc
    return x


def python_bytecode_accumulate(buffer, slot, iters, seed):
    """Q5 baseline: identical update in pure CPython bytecode (holds the GIL)."""
    m = 0xFFFFFFFFFFFFFFFF
    x = seed
    acc = int(buffer[slot])
    for _ in range(iters):
        x = (x + 0x9E3779B97F4A7C15) & m
        acc = (acc + (x >> 58) + 1) & m
        buffer[slot] = acc
    return x


# ------------------------------------------------------------------------------
# Buffers, timing and context-switch telemetry
# ------------------------------------------------------------------------------
def aligned_uint64_buffer(n_elems: int, align: int = BUFFER_ALIGN):
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) % align
    buf = raw[offset:offset + n_elems * 8].view(np.uint64)
    assert buf.ctypes.data % align == 0
    return raw, buf


def ctx_switches():
    """(voluntary, involuntary) context switches of this process so far."""
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_nvcsw), int(ru.ru_nivcsw), "resource.getrusage(RUSAGE_SELF)"
    if psutil is not None:
        c = psutil.Process().num_ctx_switches()
        return int(c.voluntary), int(c.involuntary), "psutil.num_ctx_switches (Windows: involuntary not separated)"
    return None, None, "unavailable"


def run_threads(ctx, kernel, buffer, slots, iters, n_threads, extra_args=(), pin=True):
    """Start n_threads workers, release them together, return wall ns and per-thread telemetry."""
    barrier = threading.Barrier(n_threads + 1)
    done = threading.Barrier(n_threads + 1)   # keeps workers alive until the closing ctx-switch snapshot
    per = [None] * n_threads

    def body(t):
        pinned = pin_current_thread(ctx["pin_cpus"][t]) if pin else False
        barrier.wait()
        t0 = time.perf_counter_ns()
        seed = ctx["thread_seeds"][t]
        kernel(buffer, slots[t], iters, seed if kernel is python_bytecode_accumulate else np.uint64(seed),
               *extra_args)
        per[t] = {"thread": t, "slot": int(slots[t]), "cpu": ctx["pin_cpus"][t] if pinned else None,
                  "elapsed_ns": time.perf_counter_ns() - t0}
        done.wait()

    threads = [threading.Thread(target=body, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    v0, i0, _ = ctx_switches()
    barrier.wait()
    t0 = time.perf_counter_ns()
    done.wait()
    wall = time.perf_counter_ns() - t0
    v1, i1, src = ctx_switches()       # workers still alive here, so Windows still counts them
    for th in threads:
        th.join()
    cs = {"voluntary": None if v0 is None else v1 - v0,
          "involuntary": None if i0 is None else i1 - i0, "source": src}
    return wall, per, cs


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx, iters, trials):
    n = NUM_THREADS
    raw, buf = aligned_uint64_buffer(n * max(STRIDES) + 16)
    results = {}
    finals = {}
    zero = np.uint64(0)
    for stride in STRIDES:
        slots = [np.uint64(t * stride) for t in range(n)]
        walls, per_runs, cs_runs = [], [], []
        for _ in range(trials):
            buf[:] = 0
            wall, per, cs = run_threads(ctx, worker_strided_accumulate, buf, slots, np.uint64(iters), n,
                                        extra_args=(zero,))
            walls.append(wall)
            per_runs.append(per)
            cs_runs.append(cs)
            finals.setdefault(f"stride_{stride}", []).append([int(buf[int(s)]) for s in slots])
        med = int(statistics.median(walls))
        byte_offsets = [t * stride * 8 for t in range(n)]
        line = ctx["cache_line_size"]
        results[f"stride_{stride}"] = {
            "stride_elems": stride,
            "byte_offsets": byte_offsets,
            "cache_lines_touched": sorted({o // line for o in byte_offsets}),
            "wall_ns_trials": walls,
            "wall_ns_median": med,
            "wall_ns_min": min(walls),
            "ns_per_update_per_thread": med / iters,
            "aggregate_updates_per_ns": n * iters / med,
            "per_thread_ns_median_trial": per_runs[walls.index(sorted(walls)[len(walls) // 2])],
            "ctx_switches_per_trial": cs_runs,
        }
    return results, finals, raw


def run_private_baseline(ctx, iters, trials):
    n = NUM_THREADS
    raw, buf = aligned_uint64_buffer(n * max(STRIDES) + 16)
    slots = [np.uint64(t) for t in range(n)]          # stride = 1: final stores share one line
    walls, cs_runs, finals = [], [], []
    for _ in range(trials):
        buf[:] = 0
        wall, per, cs = run_threads(ctx, worker_private_accumulate, buf, slots, np.uint64(iters), n)
        walls.append(wall)
        cs_runs.append(cs)
        finals.append([int(buf[t]) for t in range(n)])
    med = int(statistics.median(walls))
    return {"layout": "register-private accumulator, one final store per thread (slots 0..3, stride 1)",
            "wall_ns_trials": walls, "wall_ns_median": med, "wall_ns_min": min(walls),
            "ns_per_update_per_thread": med / iters, "ctx_switches_per_trial": cs_runs}, finals


def run_single_thread_reference(ctx, iters, trials):
    """One strided worker alone: the uncontended L1 store-forwarding cost per update."""
    raw, buf = aligned_uint64_buffer(32)
    walls = []
    for _ in range(trials):
        buf[:] = 0
        wall, _, _ = run_threads(ctx, worker_strided_accumulate, buf, [np.uint64(0)], np.uint64(iters), 1,
                                 extra_args=(np.uint64(0),))
        walls.append(wall)
    med = int(statistics.median(walls))
    return {"wall_ns_trials": walls, "wall_ns_median": med, "ns_per_update": med / iters}


def run_naive_promoted(ctx, iters, trials):
    """Q4 control: the promotable loop under the worst layout (stride 1)."""
    raw, buf = aligned_uint64_buffer(32)
    slots = [np.uint64(t) for t in range(NUM_THREADS)]
    walls, finals = [], []
    for _ in range(trials):
        buf[:] = 0
        wall, _, _ = run_threads(ctx, naive_promotable_accumulate, buf, slots, np.uint64(iters), NUM_THREADS)
        walls.append(wall)
        finals.append([int(buf[t]) for t in range(NUM_THREADS)])
    med = int(statistics.median(walls))
    return {"wall_ns_trials": walls, "wall_ns_median": med, "ns_per_update_per_thread": med / iters}, finals


def run_gil_baseline(ctx, iters):
    """Q5: pure-Python threads, stride 1 vs stride 8, T = 1, 2, 4 (GIL serialises the bytecode)."""
    out = {}
    for stride in (1, 8):
        for t_count in (1, 2, 4):
            raw, buf = aligned_uint64_buffer(NUM_THREADS * 8 + 16)
            slots = [t * stride for t in range(t_count)]
            wall, _, cs = run_threads(ctx, python_bytecode_accumulate, buf, slots, iters, t_count, pin=False)
            out[f"stride_{stride}_T{t_count}"] = {
                "threads": t_count, "stride": stride, "wall_ns": wall,
                "ns_per_update_aggregate": wall / (t_count * iters),
                "ctx_switches": cs,
            }
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def _native_body(ir: str, fn_marker: str) -> str:
    """Text of the compiled native function (not the cpython/cfunc boxing wrappers)."""
    for f in ir.split("\ndefine ")[1:]:
        head = f.split("(", 1)[0]
        if fn_marker in head and "cpython" not in head and "cfunc" not in head:
            return f
    return ""


def _loop_blocks(ir: str, fn_marker: str):
    """Return the self-looping basic blocks (loop bodies) of the native function named fn_marker."""
    body = _native_body(ir, fn_marker)
    blocks, label, lines = [], "entry", []
    label_re = re.compile(r'^"?([\w.$-]+)"?:(\s*;.*)?$')
    for line in body.splitlines():
        s = line.strip()
        m = label_re.match(s)
        if m:
            blocks.append((label, lines))
            label, lines = m.group(1), []
        else:
            lines.append(s)
    blocks.append((label, lines))
    loops = []
    for label, lines in blocks:
        term = next((l for l in reversed(lines) if l.startswith("br ")), "")
        if re.search(r'label %"?' + re.escape(label) + r'"?(,|\s|$)', term):
            loops.append((label, lines))
    return loops


def _ir_text(dispatcher):
    ir = dispatcher.inspect_llvm()
    return "\n".join(ir.values()) if isinstance(ir, dict) else str(ir)


def audit_llvm_and_parity(ctx, finals_strided, finals_private, finals_naive):
    # ---- parity: every layout, every trial, every thread must equal the privatized result
    reference = finals_private[0]
    parity = {}
    for layout, runs in finals_strided.items():
        parity[layout] = all(r == reference for r in runs)
    parity["private"] = all(r == reference for r in finals_private)
    parity["naive_promoted"] = all(r == reference for r in finals_naive)
    parity_ok = all(parity.values())

    # ---- LLVM IR
    report = {}
    for name, disp in (("worker_strided_accumulate", worker_strided_accumulate),
                       ("naive_promotable_accumulate", naive_promotable_accumulate),
                       ("worker_private_accumulate", worker_private_accumulate)):
        ir = _ir_text(disp)
        loops = _loop_blocks(ir, name)
        loop_lines = [l for _, ls in loops for l in ls]
        report[name] = {
            "ir_chars": len(ir),
            "store_i64_total": sum(1 for l in ir.splitlines() if l.strip().startswith("store i64")),
            "load_i64_total": sum(1 for l in ir.splitlines() if "= load i64" in l),
            "loop_blocks": [lab for lab, _ in loops],
            "store_i64_in_loop": sum(1 for l in loop_lines if l.startswith("store i64")),
            "load_i64_in_loop": sum(1 for l in loop_lines if "= load i64" in l),
            "volatile_or_atomic_in_loop": sum(1 for l in loop_lines if "volatile" in l or "atomic" in l),
            "native_calls": sorted(set(re.findall(r'call [^@\n]*@"?([\w.$]+)', _native_body(ir, name)))),
            "pyobject_fallback": bool(re.search(r'@"?(Py|_Py|numba_unpickle|numba_gil)', _native_body(ir, name))),
            "loop_excerpt": [l for l in loop_lines if l.startswith("store i64") or "= load i64" in l
                             or l.startswith("br ")][:12],
        }
        with open(f"llvm_{name}.ll", "w", encoding="utf-8") as fh:
            fh.write(ir)
    s = report["worker_strided_accumulate"]
    llvm_ok = (s["store_i64_in_loop"] >= 1 and s["load_i64_in_loop"] >= 1 and not s["pyobject_fallback"]
               and not report["worker_private_accumulate"]["pyobject_fallback"])
    return {"parity": parity, "parity_ok": parity_ok, "reference_final_counters": reference,
            "llvm": report, "llvm_ok": llvm_ok}


def main(quick: bool):
    iters = 2_000_000 if quick else ITERATIONS_PER_THREAD
    trials = 2 if quick else TRIALS
    ctx = init_system_context(GIT_USERNAME)

    print("=" * 78)
    print("Cache Line & False Sharing Mechanics - Numba nogil threads" + ("  [QUICK MODE]" if quick else ""))
    print("=" * 78)
    print(f"Git identity        : {ctx['git_user']}")
    print(f"CPU                 : {ctx['cpu_model']}")
    print(f"cpu_sig             : {ctx['cpu_sig']}")
    print(f"Logical / physical  : {ctx['logical_cores']} / {ctx['physical_cores']}")
    print(f"Cache line size     : {ctx['cache_line_size']} B")
    print(f"INSTANCE_SEED       : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    print(f"Affinity            : supported={ctx['affinity_supported']} pin_cpus={ctx['pin_cpus']} "
          f"(one logical CPU per physical core)")
    print(f"Power state         : {ctx['power']}")
    print(f"Python / NumPy / Numba : {platform.python_version()} / {np.__version__} / "
          f"{__import__('numba').__version__}  GIL enabled={getattr(sys, '_is_gil_enabled', lambda: True)()}")
    print(f"Workload            : {NUM_THREADS} threads x {iters:,} updates, {trials} trials per layout (median)")

    # JIT warm-up (compile all kernels for the uint64 signatures used below)
    _, wb = aligned_uint64_buffer(32)
    worker_strided_accumulate(wb, np.uint64(0), np.uint64(10), np.uint64(1), np.uint64(0))
    naive_promotable_accumulate(wb, np.uint64(0), np.uint64(10), np.uint64(1))
    worker_private_accumulate(wb, np.uint64(0), np.uint64(10), np.uint64(1))

    single = run_single_thread_reference(ctx, iters, trials)
    strided, finals_strided, _ = run_stride_benchmark(ctx, iters, trials)
    private, finals_private = run_private_baseline(ctx, iters, trials)
    naive, finals_naive = run_naive_promoted(ctx, iters, trials)
    gil = run_gil_baseline(ctx, GIL_ITERATIONS if not quick else 50_000)
    audit = audit_llvm_and_parity(ctx, finals_strided, finals_private, finals_naive)

    s8 = strided["stride_8"]["wall_ns_median"]
    print("\n[Task 3/4] Wall-clock medians (4 concurrent threads unless noted)")
    print(f"  {'layout':34s} {'median ns':>15s} {'ns/update':>10s} {'vs stride 8':>12s}   lines touched")
    print(f"  {'single thread (stride n/a)':34s} {single['wall_ns_median']:15,d} {single['ns_per_update']:10.3f} "
          f"{single['wall_ns_median'] / s8:11.3f}x")
    for k, v in strided.items():
        label = f"{k} ({v['stride_elems'] * 8} B spacing)"
        print(f"  {label:34s} {v['wall_ns_median']:15,d} "
              f"{v['ns_per_update_per_thread']:10.3f} {v['wall_ns_median'] / s8:11.3f}x   {v['cache_lines_touched']}")
    print(f"  {'register-private (Task 4)':34s} {private['wall_ns_median']:15,d} "
          f"{private['ns_per_update_per_thread']:10.3f} {private['wall_ns_median'] / s8:11.3f}x")
    print(f"  {'naive promotable loop, stride 1':34s} {naive['wall_ns_median']:15,d} "
          f"{naive['ns_per_update_per_thread']:10.3f} {naive['wall_ns_median'] / s8:11.3f}x")
    s1 = strided["stride_1"]["wall_ns_median"]
    print(f"\n  slowdown stride 1 / stride 8  = {s1 / s8:.3f}x")
    print(f"  slowdown stride 4 / stride 8  = {strided['stride_4']['wall_ns_median'] / s8:.3f}x")
    print(f"  stride 16 / stride 8          = {strided['stride_16']['wall_ns_median'] / s8:.3f}x")
    print(f"  stride 8 / private register   = {s8 / private['wall_ns_median']:.3f}x")
    print("\n  Context switches per trial (voluntary/involuntary):")
    for k, v in list(strided.items()) + [("private", private)]:
        cs = v["ctx_switches_per_trial"]
        print(f"    {k:10s} " + "  ".join(f"{c['voluntary']}/{c['involuntary']}" for c in cs))
    print(f"    source: {strided['stride_1']['ctx_switches_per_trial'][0]['source']}")

    print("\n  Per-thread elapsed ns (median trial):")
    for k, v in strided.items():
        print(f"    {k:10s} " + "  ".join(f"T{p['thread']}@cpu{p['cpu']}={p['elapsed_ns']:,}"
                                          for p in v["per_thread_ns_median_trial"]))

    print("\n[Q5] Pure-Python bytecode threads (GIL held), per-thread iterations "
          f"{GIL_ITERATIONS if not quick else 50_000:,}")
    for k, v in gil.items():
        print(f"  {k:14s} wall {v['wall_ns']:15,d} ns   aggregate {v['ns_per_update_aggregate']:9.2f} ns/update")

    print("\n[Task 5] Parity (final counters must equal the register-private reference)")
    print(f"  reference counters : {audit['reference_final_counters']}")
    for k, v in audit["parity"].items():
        print(f"  {k:16s} {'OK' if v else 'MISMATCH'}")
    print("\n[Task 5] LLVM IR audit (inspect_llvm)")
    for name, r in audit["llvm"].items():
        print(f"  {name}: loop blocks {r['loop_blocks']}  store i64 in loop={r['store_i64_in_loop']} "
              f"load i64 in loop={r['load_i64_in_loop']}  (total store/load {r['store_i64_total']}/"
              f"{r['load_i64_total']})  pyobject={r['pyobject_fallback']}  calls={r['native_calls']}")
        for l in r["loop_excerpt"]:
            print(f"      {l}")

    gil_released = strided["stride_16"]["wall_ns_median"] < 2 * single["wall_ns_median"]
    verification_passed = bool(audit["parity_ok"] and audit["llvm_ok"] and gil_released)
    print(f"\n  GIL released (4-thread stride 16 < 2x single thread): {gil_released}")
    print(f"  verification_passed: {verification_passed}")

    manifest = {
        "git_username": ctx["git_user"],
        "mode": "quick" if quick else "full",
        "cpu_model": ctx["cpu_model"],
        "cpu_sig": ctx["cpu_sig"],
        "platform": ctx["platform"],
        "logical_cores": ctx["logical_cores"],
        "physical_cores": ctx["physical_cores"],
        "core_groups": ctx["core_groups"],
        "cache_line_size": ctx["cache_line_size"],
        "instance_seed": ctx["instance_seed"],
        "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
        "thread_seeds": ctx["thread_seeds"],
        "affinity_supported": ctx["affinity_supported"],
        "pin_cpus": ctx["pin_cpus"],
        "power": ctx["power"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": iters,
        "trials": trials,
        "buffer_alignment_bytes": BUFFER_ALIGN,
        "single_thread_reference": single,
        "stride_results": strided,
        "private_baseline": private,
        "naive_promoted_stride1": naive,
        "slowdown_stride1_vs_stride8": s1 / s8,
        "slowdown_stride4_vs_stride8": strided["stride_4"]["wall_ns_median"] / s8,
        "ratio_stride16_vs_stride8": strided["stride_16"]["wall_ns_median"] / s8,
        "ratio_stride8_vs_private": s8 / private["wall_ns_median"],
        "gil_bytecode_baseline": gil,
        "parity": audit["parity"],
        "reference_final_counters": audit["reference_final_counters"],
        "llvm_audit": audit["llvm"],
        "gil_released": gil_released,
        "verification_passed": verification_passed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": __import__("numba").__version__},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.abspath(__file__), "rb") as fh:
        manifest["script_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  {MANIFEST_PATH} written (sha256 {manifest['manifest_sha256'][:16]}...)")
    return 0 if verification_passed else 1


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
