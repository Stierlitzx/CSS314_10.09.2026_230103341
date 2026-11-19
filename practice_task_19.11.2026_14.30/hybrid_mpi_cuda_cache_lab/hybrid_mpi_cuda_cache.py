import os
import sys
import json
import math
import time
import re
import hashlib
import platform
import statistics
import subprocess
import warnings
import numpy as np

# Small ring launches are intentional; silence Numba's occupancy hint and deprecation chatter.
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

# Additional constants (recorded in the manifest)
DIAG_EVERY = 100
CFL_SAFETY = 0.90              # dt = CFL_SAFETY * dx^2 / (4 kappa_max)
WARMUP_STEPS = 3
PACK_THREADS = 128
SHM_SLOTS = 64                 # uint64 slots exposed by the shared window
SHM_ALIGN = 128                # 128 B base => 64 B aligned and on a 128 B prefetch-pair boundary
SHM_STRIDES = (1, 8, 16)
SHM_TRIALS = 3
MANIFEST_PATH = "hybrid_lab_manifest.json"

TAG_TO_UP, TAG_TO_DOWN, TAG_TO_LEFT, TAG_TO_RIGHT = 10, 11, 12, 13
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_GAMMA, _S58, _ONE = np.uint64(0x9E3779B97F4A7C15), np.uint64(58), np.uint64(1)


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def get_gpu_name() -> str:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                                  encoding="utf-8")
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def cpu_model_string() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                           "(Get-CimInstance Win32_Processor).Name"],
                                          encoding="utf-8", stderr=subprocess.DEVNULL)
            return out.strip().splitlines()[0].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def current_affinity():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def log0(ctx, *args):
    if ctx["rank"] == 0:
        print(*args, flush=True)


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")
    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    gpu_name = comm.bcast(get_gpu_name() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # ---- Tier 1: non-periodic 2x2 Cartesian communicator
    dims, periods = [2, 2], [False, False]
    cart_comm = comm.Create_cart(dims=dims, periods=periods, reorder=False)
    coords = tuple(int(c) for c in cart_comm.Get_coords(cart_comm.Get_rank()))
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    # ---- Tier 4: node-local communicator and MPI-3 shared window (rank 0 of the node owns the segment)
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    alloc_bytes = (SHM_SLOTS * 8 + SHM_ALIGN) if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(alloc_bytes, 8, comm=node_comm)
    buf, itemsize = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint8, shape=(SHM_SLOTS * 8 + SHM_ALIGN,))
    offset = (-raw.ctypes.data) % SHM_ALIGN
    shm_view = raw[offset:offset + SHM_SLOTS * 8].view(np.uint64)
    assert shm_view.ctypes.data % 64 == 0
    win.Lock_all(MPI.MODE_NOCHECK)      # passive-target epoch for direct load/store access

    ny, nx = GLOBAL_NY // dims[0], GLOBAL_NX // dims[1]
    return {
        "comm": comm, "cart_comm": cart_comm, "node_comm": node_comm, "win": win, "shm_view": shm_view,
        "rank": cart_comm.Get_rank(), "size": size, "node_rank": node_rank, "node_size": node_comm.Get_size(),
        "dims": dims, "periods": periods, "coords": coords,
        "nbr_up": nbr_up, "nbr_down": nbr_down, "nbr_left": nbr_left, "nbr_right": nbr_right,
        "ny": ny, "nx": nx,
        "github_user": github_user.strip(), "gpu_uuid": gpu_uuid, "gpu_name": gpu_name,
        "cpu_sig": cpu_sig, "cpu_model": cpu_model_string(), "cpu_line_bytes": line_bytes,
        "instance_seed": instance_seed,
        "shm_base_mod64": int(shm_view.ctypes.data % 64), "shm_base_mod128": int(shm_view.ctypes.data % 128),
        "shm_disp_unit": int(itemsize), "affinity": current_affinity(), "pid": os.getpid(),
    }


def rank_table(ctx):
    """Per-rank metadata assembled with Allreduce (no field arrays are ever collected)."""
    pn = MPI.PROC_NULL
    row = np.zeros((ctx["size"], 10), dtype=np.float64)
    nbrs = [ctx["nbr_up"], ctx["nbr_down"], ctx["nbr_left"], ctx["nbr_right"]]
    row[ctx["rank"]] = [ctx["coords"][0], ctx["coords"][1], *[(-1 if n == pn else n) for n in nbrs],
                        ctx["ny"] + 2, ctx["nx"] + 2, ctx["node_rank"], ctx["pid"]]
    table = np.zeros_like(row)
    ctx["cart_comm"].Allreduce(row, table, op=MPI.SUM)
    return table


# ==============================================================================
# TASK 2: Rank-Local GPU Subdomain & Halo Pack/Unpack Kernels
# ==============================================================================
def seeded_parameters(instance_seed: int):
    rng = np.random.Generator(np.random.PCG64(instance_seed))
    n_pulses = 3
    return {
        "pulse_cx": rng.uniform(0.22, 0.78, n_pulses).tolist(),
        "pulse_cy": rng.uniform(0.22, 0.78, n_pulses).tolist(),
        "pulse_amp": rng.uniform(0.60, 1.40, n_pulses).tolist(),
        "pulse_sigma": rng.uniform(0.040, 0.070, n_pulses).tolist(),
        "kappa_fx": int(rng.integers(1, 4)),
        "kappa_fy": int(rng.integers(1, 4)),
        "kappa_phx": float(rng.uniform(0.0, 1.0)),
        "kappa_phy": float(rng.uniform(0.0, 1.0)),
        "src_amp": float(rng.uniform(3.0, 6.0)),
        "src_sigma": 0.030,
        "src_pos": [float(v) for v in rng.uniform(0.30, 0.70, 2)],
        "sink_pos": [float(v) for v in rng.uniform(0.30, 0.70, 2)],
    }


def initialize_fields(ctx, params):
    """Rank-local (ny+2, nx+2) u, kappa, S from analytic global coordinates; no global grid exists.

    Local index li maps to global gi = row_rank*ny + li and y = gi/513, so gi = 0 and 513 are the
    Dirichlet walls and 1..512 are interior cells (dx = dy = 1/513).
    """
    ny, nx = ctx["ny"], ctx["nx"]
    rr, cr = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    gi = rr * ny + np.arange(ny + 2, dtype=np.float64)
    gj = cr * nx + np.arange(nx + 2, dtype=np.float64)
    Y, X = np.meshgrid(gi * dy, gj * dx, indexing="ij")
    p = params
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * (p["kappa_fx"] * X + p["kappa_phx"])) \
                       * np.cos(2.0 * np.pi * (p["kappa_fy"] * Y + p["kappa_phy"]))
    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    for cx, cy, a, s in zip(p["pulse_cx"], p["pulse_cy"], p["pulse_amp"], p["pulse_sigma"]):
        u += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * s * s))
    s2 = 2.0 * p["src_sigma"] ** 2
    src = p["src_amp"] * (np.exp(-((X - p["src_pos"][0]) ** 2 + (Y - p["src_pos"][1]) ** 2) / s2)
                          - np.exp(-((X - p["sink_pos"][0]) ** 2 + (Y - p["sink_pos"][1]) ** 2) / s2))
    if ctx["nbr_up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0
    return {"u": np.ascontiguousarray(u), "kappa": np.ascontiguousarray(kappa),
            "src": np.ascontiguousarray(src), "dx": dx, "dy": dy}


@cuda.jit
def pack_halos_kernel(u, send_buf, ny, nx):
    """Gather the owned border strips (two of them strided columns) into contiguous rows."""
    k = cuda.grid(1)
    if k < nx:
        send_buf[0, k] = u[1, k + 1]       # top row     -> up
        send_buf[1, k] = u[ny, k + 1]      # bottom row  -> down
    if k < ny:
        send_buf[2, k] = u[k + 1, 1]       # left column  (stride nx+2) -> left
        send_buf[3, k] = u[k + 1, nx]      # right column (stride nx+2) -> right


@cuda.jit
def unpack_halos_kernel(u, recv_buf, ny, nx, has_up, has_down, has_left, has_right):
    """Scatter received vectors into ghost rows/columns 0 and n+1 (= 257)."""
    k = cuda.grid(1)
    if k < nx:
        if has_up:
            u[0, k + 1] = recv_buf[0, k]
        if has_down:
            u[ny + 1, k + 1] = recv_buf[1, k]
    if k < ny:
        if has_left:
            u[k + 1, 0] = recv_buf[2, k]
        if has_right:
            u[k + 1, nx + 1] = recv_buf[3, k]


def allocate_exchange_buffers(ctx):
    n = max(ctx["ny"], ctx["nx"])
    bufs = {"d_send": cuda.device_array((4, n), dtype=np.float64),
            "d_recv": cuda.to_device(np.zeros((4, n), dtype=np.float64)),
            "h_send": cuda.pinned_array((4, n), dtype=np.float64),
            "h_recv": cuda.pinned_array((4, n), dtype=np.float64)}
    bufs["h_send"][:] = 0.0
    bufs["h_recv"][:] = 0.0
    return bufs


def exchange_halos_async(ctx, bufs, d_u, stream):
    """GPU pack -> async D2H into pinned memory -> post tag-matched Irecv/Isend; returns requests."""
    ny, nx, cart = ctx["ny"], ctx["nx"], ctx["cart_comm"]
    n = bufs["d_send"].shape[1]
    pack_halos_kernel[(n + PACK_THREADS - 1) // PACK_THREADS, PACK_THREADS, stream](d_u, bufs["d_send"], ny, nx)
    bufs["d_send"].copy_to_host(bufs["h_send"], stream=stream)
    stream.synchronize()
    hs, hr = bufs["h_send"], bufs["h_recv"]
    return [
        cart.Irecv([hr[UP, :nx], MPI.DOUBLE], source=ctx["nbr_up"], tag=TAG_TO_DOWN),
        cart.Irecv([hr[DOWN, :nx], MPI.DOUBLE], source=ctx["nbr_down"], tag=TAG_TO_UP),
        cart.Irecv([hr[LEFT, :ny], MPI.DOUBLE], source=ctx["nbr_left"], tag=TAG_TO_RIGHT),
        cart.Irecv([hr[RIGHT, :ny], MPI.DOUBLE], source=ctx["nbr_right"], tag=TAG_TO_LEFT),
        cart.Isend([hs[UP, :nx], MPI.DOUBLE], dest=ctx["nbr_up"], tag=TAG_TO_UP),
        cart.Isend([hs[DOWN, :nx], MPI.DOUBLE], dest=ctx["nbr_down"], tag=TAG_TO_DOWN),
        cart.Isend([hs[LEFT, :ny], MPI.DOUBLE], dest=ctx["nbr_left"], tag=TAG_TO_LEFT),
        cart.Isend([hs[RIGHT, :ny], MPI.DOUBLE], dest=ctx["nbr_right"], tag=TAG_TO_RIGHT),
    ]


def complete_halo_exchange(ctx, bufs, d_u, reqs, stream):
    """Waitall -> async H2D of pinned halos -> GPU unpack into ghost cells."""
    MPI.Request.Waitall(reqs)
    bufs["d_recv"].copy_to_device(bufs["h_recv"], stream=stream)
    n = bufs["d_recv"].shape[1]
    pn = MPI.PROC_NULL
    unpack_halos_kernel[(n + PACK_THREADS - 1) // PACK_THREADS, PACK_THREADS, stream](
        d_u, bufs["d_recv"], ctx["ny"], ctx["nx"],
        int(ctx["nbr_up"] != pn), int(ctx["nbr_down"] != pn),
        int(ctx["nbr_left"] != pn), int(ctx["nbr_right"] != pn))


# ==============================================================================
# TASK 3: 2D Shared-Memory Tiled Stencil & Dual-Stream Pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, dt, inv_dx2, inv_dy2, lam,
                         r_start, r_end, c_start, c_end):
    """Update u_new[r_start..r_end, c_start..c_end] with an 18x18 shared-memory tile.

    Reads stay inside the cross-shaped neighbourhood of the box, so the deep-interior launch
    [2..n-1]^2 never touches ghost cells that stream_boundary is unpacking concurrently.
    """
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    sy = ty + 1
    sx = tx + 1
    if i <= r_end + 1 and j <= c_end + 1:
        smem[sy, sx] = u[i, j]
    if ty == 0 and j <= c_end:
        smem[0, sx] = u[i - 1, j]
    if ty == TILE_DIM - 1 and i + 1 <= r_end + 1 and j <= c_end:
        smem[SHARED_DIM - 1, sx] = u[i + 1, j]
    if tx == 0 and i <= r_end:
        smem[sy, 0] = u[i, j - 1]
    if tx == TILE_DIM - 1 and j + 1 <= c_end + 1 and i <= r_end:
        smem[sy, SHARED_DIM - 1] = u[i, j + 1]

    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        c = smem[sy, sx]
        kc = kappa[i, j]
        k_n = 0.5 * (kappa[i - 1, j] + kc)
        k_s = 0.5 * (kappa[i + 1, j] + kc)
        k_w = 0.5 * (kappa[i, j - 1] + kc)
        k_e = 0.5 * (kappa[i, j + 1] + kc)
        div_y = (k_s * (smem[sy + 1, sx] - c) - k_n * (c - smem[sy - 1, sx])) * inv_dy2
        div_x = (k_e * (smem[sy, sx + 1] - c) - k_w * (c - smem[sy, sx - 1])) * inv_dx2
        u_new[i, j] = c + dt * (div_x + div_y - lam * c * c * c + src[i, j])


def launch_stencil(state, stream, r0, r1, c0, c1):
    grid = ((c1 - c0 + TILE_DIM) // TILE_DIM, (r1 - r0 + TILE_DIM) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        state["d_u"], state["d_u_new"], state["d_kappa"], state["d_src"],
        state["dt"], state["inv_dx2"], state["inv_dy2"], LAMBDA_DAMP, r0, r1, c0, c1)


def step_serialized(ctx, state, bufs, streams, timers):
    s = streams["boundary"]
    ny, nx = ctx["ny"], ctx["nx"]
    e0, e1 = timers["ev"][0], timers["ev"][1]
    e0.record(s)
    reqs = exchange_halos_async(ctx, bufs, state["d_u"], s)
    complete_halo_exchange(ctx, bufs, state["d_u"], reqs, s)
    launch_stencil(state, s, 1, ny, 1, nx)
    e1.record(s)
    s.synchronize()
    timers["full_ms"] += cuda.event_elapsed_time(e0, e1)
    state["d_u"], state["d_u_new"] = state["d_u_new"], state["d_u"]


def step_overlapped(ctx, state, bufs, streams, timers):
    sb, si = streams["boundary"], streams["interior"]
    ny, nx = ctx["ny"], ctx["nx"]
    ev = timers["ev"]
    ev[0].record(sb)
    reqs = exchange_halos_async(ctx, bufs, state["d_u"], sb)          # pack + D2H + post Irecv/Isend
    ev[1].record(sb)
    ev[2].record(si)
    launch_stencil(state, si, 2, ny - 1, 2, nx - 1)                    # deep interior [2..255]^2
    ev[3].record(si)
    ev[4].record(sb)
    complete_halo_exchange(ctx, bufs, state["d_u"], reqs, sb)         # Waitall + H2D + unpack
    launch_stencil(state, sb, 1, 1, 1, nx)                             # 4 outer border strips
    launch_stencil(state, sb, ny, ny, 1, nx)
    launch_stencil(state, sb, 2, ny - 1, 1, 1)
    launch_stencil(state, sb, 2, ny - 1, nx, nx)
    ev[5].record(sb)
    sb.synchronize()
    si.synchronize()
    timers["pack_ms"] += cuda.event_elapsed_time(ev[0], ev[1])
    timers["interior_ms"] += cuda.event_elapsed_time(ev[2], ev[3])
    timers["boundary_ms"] += cuda.event_elapsed_time(ev[4], ev[5])
    state["d_u"], state["d_u_new"] = state["d_u_new"], state["d_u"]


def make_state(fields, dt):
    return {"d_u": cuda.to_device(fields["u"]), "d_u_new": cuda.to_device(fields["u"]),
            "d_kappa": cuda.to_device(fields["kappa"]), "d_src": cuda.to_device(fields["src"]),
            "dt": float(dt), "inv_dx2": 1.0 / fields["dx"] ** 2, "inv_dy2": 1.0 / fields["dy"] ** 2}


def global_diagnostics(ctx, state, fields, step):
    """Global energy E = sum(u dx dy), L2 step residual and max|u| via cart_comm.allreduce."""
    ny, nx = ctx["ny"], ctx["nx"]
    u = state["d_u"].copy_to_host()[1:ny + 1, 1:nx + 1]
    u_prev = state["d_u_new"].copy_to_host()[1:ny + 1, 1:nx + 1]
    cart = ctx["cart_comm"]
    energy = cart.allreduce(float(u.sum() * fields["dx"] * fields["dy"]), op=MPI.SUM)
    res2 = cart.allreduce(0.0 if step == 0 else float(((u - u_prev) ** 2).sum()), op=MPI.SUM)
    umax = cart.allreduce(float(np.abs(u).max()), op=MPI.MAX)
    return {"step": step, "energy": energy, "l2_residual": math.sqrt(res2), "u_max": umax}


def run_mode(ctx, fields, dt, bufs, streams, mode, steps, record=True):
    state = make_state(fields, dt)
    timers = {"ev": [cuda.event(timing=True) for _ in range(6)],
              "full_ms": 0.0, "pack_ms": 0.0, "interior_ms": 0.0, "boundary_ms": 0.0}
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    trace = [global_diagnostics(ctx, state, fields, 0)] if record else []
    ctx["cart_comm"].Barrier()
    t0 = MPI.Wtime()
    for n in range(1, steps + 1):
        step_fn(ctx, state, bufs, streams, timers)
        if record and n % DIAG_EVERY == 0:
            trace.append(global_diagnostics(ctx, state, fields, n))
    ctx["cart_comm"].Barrier()
    wall_ms = (MPI.Wtime() - t0) * 1e3
    timers.pop("ev")
    return state, trace, wall_ms, timers


# ==============================================================================
# TASK 4: Multi-Rank Host Cache-Line False Sharing Sweep (MPI-3 shared window)
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iters, seed, mask):
    """In-place read-modify-write of shm_view[slot] every iteration, no register hoisting.

    The address slot + (x & mask) depends on the loop-carried Weyl state x; mask is 0 at run time,
    which the compiler cannot prove, so each iteration is a real load + store to the shared window.
    """
    x = seed
    for _ in range(iters):
        x += _GAMMA
        j = slot + (x & mask)
        shm_view[j] = shm_view[j] + (x >> _S58) + _ONE
    return x


@njit(nogil=True)
def host_private_accumulate(iters, seed):
    """Reference: same arithmetic in a register, used to verify the window counters bit-exactly."""
    x = seed
    acc = np.uint64(0)
    for _ in range(iters):
        x += _GAMMA
        acc += (x >> _S58) + _ONE
    return acc


def _win_sync(ctx):
    ctx["win"].Sync()
    ctx["node_comm"].Barrier()
    ctx["win"].Sync()


def benchmark_shm_strides(ctx, iters, trials):
    rank, node = ctx["rank"], ctx["node_comm"]
    view = ctx["shm_view"]
    seed = np.uint64((ctx["instance_seed"] + 0x9E3779B97F4A7C15 * (rank + 1)) & 0xFFFFFFFFFFFFFFFF)
    expected = int(host_private_accumulate(np.uint64(iters), seed))
    host_shm_accumulate(view, np.uint64(SHM_SLOTS - 1), np.uint64(1), seed, np.uint64(0))   # JIT warm-up
    results = {}
    for stride in SHM_STRIDES:
        slot = rank * stride
        mine, ok = [], True
        for _ in range(trials):
            if ctx["node_rank"] == 0:
                view[:] = 0
            _win_sync(ctx)
            t0 = time.perf_counter_ns()
            host_shm_accumulate(view, np.uint64(slot), np.uint64(iters), seed, np.uint64(0))
            mine.append(time.perf_counter_ns() - t0)
            # Only this rank writes view[slot], so it is final now. Reading it after the closing
            # barrier would race with node rank 0 zeroing the window for the next trial.
            ctx["win"].Sync()
            ok &= int(view[slot]) == expected
            _win_sync(ctx)
        # slowest rank per trial defines the trial time
        trial_max = [node.allreduce(v, op=MPI.MAX) for v in mine]
        trial_min = [node.allreduce(v, op=MPI.MIN) for v in mine]
        ok_all = node.allreduce(int(ok), op=MPI.MIN) == 1
        offsets = [r * stride * 8 for r in range(ctx["node_size"])]
        line = ctx["cpu_line_bytes"]
        results[f"stride_{stride}"] = {
            "stride_elems": stride, "byte_offsets": offsets,
            "cpu_lines_touched": sorted({(ctx["shm_base_mod64"] + o) // line for o in offsets}),
            "slowest_rank_ns_trials": trial_max, "fastest_rank_ns_trials": trial_min,
            "median_ns": int(statistics.median(trial_max)),
            "ns_per_update": statistics.median(trial_max) / iters,
            "counters_bit_exact": ok_all,
        }
    return results, expected


# ==============================================================================
# Audits: GPU PTX (inspect_asm) and CPU LLVM IR (inspect_llvm)
# ==============================================================================
def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", ptx)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", ptx)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", ptx)),
        "barrier.sync": len(re.findall(r"\bbarrier\.sync\b", ptx)),
    }
    passed = counts["st.shared.f64"] > 0 and counts["ld.shared.f64"] > 0 and \
        (counts["bar.sync"] + counts["barrier.sync"]) > 0
    return ptx, counts, passed


def _native_body(ir: str, marker: str) -> str:
    for f in ir.split("\ndefine ")[1:]:
        head = f.split("(", 1)[0]
        if marker in head and "cpython" not in head and "cfunc" not in head:
            return f
    return ""


def llvm_audit():
    ir = host_shm_accumulate.inspect_llvm()
    ir = "\n".join(ir.values()) if isinstance(ir, dict) else str(ir)
    body = _native_body(ir, "host_shm_accumulate")
    blocks, label, lines = [], "entry", []
    label_re = re.compile(r'^"?([\w.$-]+)"?:(\s*;.*)?$')
    for raw in body.splitlines():
        s = raw.strip()
        m = label_re.match(s)
        if m:
            blocks.append((label, lines))
            label, lines = m.group(1), []
        else:
            lines.append(s)
    blocks.append((label, lines))
    loops = [(lab, ls) for lab, ls in blocks
             if re.search(r'label %"?' + re.escape(lab) + r'"?(,|\s|$)',
                          next((l for l in reversed(ls) if l.startswith("br ")), ""))]
    loop_lines = [l for _, ls in loops for l in ls]
    report = {
        "loop_blocks": [lab for lab, _ in loops],
        "store_i64_in_loop": sum(1 for l in loop_lines if l.startswith("store i64")),
        "load_i64_in_loop": sum(1 for l in loop_lines if "= load i64" in l),
        "vector_ops_in_loop": sum(1 for l in loop_lines if "<" in l and " x i64>" in l),
        "native_calls": sorted(set(re.findall(r'call [^@\n]*@"?([\w.$]+)', body))),
        "pyobject_fallback": bool(re.search(r'@"?(Py|_Py|numba_unpickle|numba_gil)', body)),
        "loop_excerpt": [l for l in loop_lines if l.startswith("store i64") or "= load i64" in l
                         or "and i64" in l][:9],
    }
    report["passed"] = (report["store_i64_in_loop"] > 0 and report["load_i64_in_loop"] > 0
                        and not report["pyobject_fallback"])
    return ir, report


def reduce_scalar(ctx, value, op):
    return ctx["cart_comm"].allreduce(float(value), op=op)


# ==============================================================================
# TASK 5: Cross-Tier Verification & Manifest Export
# ==============================================================================
def main(github_user: str):
    ctx = init_hybrid_context(github_user)
    rank, ny, nx = ctx["rank"], ctx["ny"], ctx["nx"]
    log0(ctx, "=" * 78)
    log0(ctx, "Hybrid MPI-CUDA distributed stencil + MPI-3 shared-window cache-line coherency")
    log0(ctx, "=" * 78)
    log0(ctx, f"GitHub identity   : {ctx['github_user']}")
    log0(ctx, f"GPU               : {ctx['gpu_name']}  UUID {ctx['gpu_uuid']}")
    log0(ctx, f"CPU               : {ctx['cpu_model']}  cpu_sig={ctx['cpu_sig']}  line={ctx['cpu_line_bytes']} B")
    log0(ctx, f"INSTANCE_SEED     : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    log0(ctx, f"MPI library       : {MPI.Get_library_version().splitlines()[0].strip()}")
    log0(ctx, f"MPI ranks         : {ctx['size']} in a 2x2 Cartesian grid, periods={ctx['periods']}; "
              f"node_comm size {ctx['node_size']}")

    table = rank_table(ctx)
    log0(ctx, "\n[Task 1] Topology (PROC_NULL = '-')")
    log0(ctx, "  rank coords  up down left right  local_shape  node_rank   pid")
    for r in range(ctx["size"]):
        t = table[r].astype(int)
        f = lambda v: " -" if v < 0 else f"{v:2d}"
        log0(ctx, f"  {r:4d} ({t[0]},{t[1]})  {f(t[2])}  {f(t[3])}   {f(t[4])}    {f(t[5])}   ({t[6]}, {t[7]})"
                  f"        {t[8]}     {t[9]}")
    log0(ctx, f"  shared window: {SHM_SLOTS} uint64 slots, base % 64 = {ctx['shm_base_mod64']}, "
              f"base % 128 = {ctx['shm_base_mod128']}, disp_unit = {ctx['shm_disp_unit']}")
    aff = ctx["cart_comm"].allreduce([(rank, ctx["affinity"])], op=MPI.SUM)
    log0(ctx, f"  CPU affinity per rank: {sorted(aff)}  (os.cpu_count() = {os.cpu_count()})")

    # ---- Task 2 fields
    params = seeded_parameters(ctx["instance_seed"])
    fields = initialize_fields(ctx, params)
    kappa_max = reduce_scalar(ctx, fields["kappa"].max(), MPI.MAX)
    kappa_min = reduce_scalar(ctx, fields["kappa"].min(), MPI.MIN)
    owned = (slice(1, ny + 1), slice(1, nx + 1))
    src_integral = reduce_scalar(ctx, fields["src"][owned].sum() * fields["dx"] * fields["dy"], MPI.SUM)
    dx = fields["dx"]
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max
    log0(ctx, f"\n[Task 2] Rank-local {ny + 2}x{nx + 2} fields (no global grid)")
    log0(ctx, f"  dx = dy = 1/{GLOBAL_NX + 1} = {dx:.10e}")
    log0(ctx, f"  kappa range [{kappa_min:.6f}, {kappa_max:.6f}];  integral S dA = {src_integral:.3e}")
    log0(ctx, f"  pulses x={np.round(params['pulse_cx'], 4).tolist()} y={np.round(params['pulse_cy'], 4).tolist()} "
              f"A={np.round(params['pulse_amp'], 4).tolist()} sigma={np.round(params['pulse_sigma'], 4).tolist()}")
    log0(ctx, f"  CFL: dt_max = dx^2/(4 kappa_max) = {dt_max:.6e}; dt = {CFL_SAFETY} * dt_max = {dt:.6e}")

    # ---- Task 3: GPU tiers
    bufs = allocate_exchange_buffers(ctx)
    streams = {"boundary": cuda.stream(), "interior": cuda.stream()}
    run_mode(ctx, fields, dt, bufs, streams, "serialized", WARMUP_STEPS, record=False)
    run_mode(ctx, fields, dt, bufs, streams, "overlapped", WARMUP_STEPS, record=False)
    log0(ctx, f"\n[Task 3/5] {STENCIL_STEPS} stencil steps per mode (allreduce diagnostics every {DIAG_EVERY})")
    gpu = {}
    for mode in ("serialized", "overlapped"):
        state, trace, wall_ms, timers = run_mode(ctx, fields, dt, bufs, streams, mode, STENCIL_STEPS)
        gpu[mode] = {"state": state, "trace": trace, "wall_ms": wall_ms, "timers": timers}
        log0(ctx, f"  -- {mode} --")
        log0(ctx, "     step        energy E (allreduce)        L2 residual           max|u|")
        for d in trace:
            log0(ctx, f"     {d['step']:4d}   {d['energy']:.15e}   {d['l2_residual']:.12e}   {d['u_max']:.10f}")
    u_ser = gpu["serialized"]["state"]["d_u"].copy_to_host()
    u_ovl = gpu["overlapped"]["state"]["d_u"].copy_to_host()
    parity_max_diff = reduce_scalar(ctx, np.abs(u_ser[owned] - u_ovl[owned]).max(), MPI.MAX)
    gpu_serialized_ms = reduce_scalar(ctx, gpu["serialized"]["wall_ms"], MPI.MAX)
    gpu_overlapped_ms = reduce_scalar(ctx, gpu["overlapped"]["wall_ms"], MPI.MAX)
    ev = {"serialized_step_total_ms": gpu["serialized"]["timers"]["full_ms"],
          "overlapped_pack_d2h_ms": gpu["overlapped"]["timers"]["pack_ms"],
          "overlapped_interior_kernel_ms": gpu["overlapped"]["timers"]["interior_ms"],
          "overlapped_boundary_stageC_ms": gpu["overlapped"]["timers"]["boundary_ms"]}
    cuda_events = {k: {"mean_over_ranks": reduce_scalar(ctx, v, MPI.SUM) / ctx["size"],
                       "max_over_ranks": reduce_scalar(ctx, v, MPI.MAX)} for k, v in ev.items()}
    energies = [d["energy"] for d in gpu["overlapped"]["trace"]]
    energy_monotonic = all(b <= a for a, b in zip(energies, energies[1:]))
    log0(ctx, f"  parity_max_diff = {parity_max_diff:.3e} -> {'PASS' if parity_max_diff < 1e-12 else 'FAIL'}")
    log0(ctx, f"  gpu_serialized_ms = {gpu_serialized_ms:.3f}   gpu_overlapped_ms = {gpu_overlapped_ms:.3f}   "
              f"speed-up = {gpu_serialized_ms / gpu_overlapped_ms:.4f}x")
    for k, v in cuda_events.items():
        log0(ctx, f"    CUDA event {k:32s} {v['mean_over_ranks']:10.3f} ms "
                  f"({v['mean_over_ranks'] / STENCIL_STEPS * 1e3:8.2f} us/step, max rank {v['max_over_ranks']:.3f} ms)")
    log0(ctx, f"  energy monotonic non-increasing: {energy_monotonic}")

    # ---- Task 4: host cache-line sweep on the MPI-3 shared window
    log0(ctx, f"\n[Task 4] MPI-3 shared-window false-sharing sweep: {ctx['node_size']} ranks x {SHM_ITERS:,} "
              f"updates, {SHM_TRIALS} trials, slowest rank per trial")
    shm, expected = benchmark_shm_strides(ctx, SHM_ITERS, SHM_TRIALS)
    s8 = shm["stride_8"]["median_ns"]
    for k, v in shm.items():
        log0(ctx, f"  {k:10s} offsets {str(v['byte_offsets']):18s} lines {str(v['cpu_lines_touched']):14s} "
                  f"median {v['median_ns']:>13,d} ns  {v['ns_per_update']:.3f} ns/update  "
                  f"vs stride 8 {v['median_ns'] / s8:.3f}x  bit-exact={v['counters_bit_exact']}")
        log0(ctx, f"             slowest-rank trials {v['slowest_rank_ns_trials']}  fastest-rank trials "
                  f"{v['fastest_rank_ns_trials']}")
    shm_slowdown = shm["stride_1"]["median_ns"] / s8
    log0(ctx, f"  false-sharing slowdown stride 1 / stride 8 = {shm_slowdown:.3f}x;  "
              f"stride 16 / stride 8 = {shm['stride_16']['median_ns'] / s8:.3f}x")

    # ---- Audits (rank 0) + manifest
    if rank == 0:
        ptx, ptx_counts, ptx_ok = ptx_audit()
        ir, llvm = llvm_audit()
        with open("tiled_stencil_kernel.ptx", "w", encoding="utf-8") as fh:
            fh.write(ptx)
        with open("host_shm_accumulate.ll", "w", encoding="utf-8") as fh:
            fh.write(ir)
        log0(ctx, "\n[Task 5] GPU PTX audit (tiled_stencil_kernel.inspect_asm)")
        for k, v in ptx_counts.items():
            log0(ctx, f"  {k:14s}: {v}")
        log0(ctx, f"  ptx_audit_passed: {ptx_ok}")
        log0(ctx, "[Task 5] CPU LLVM IR audit (host_shm_accumulate.inspect_llvm)")
        log0(ctx, f"  loop blocks {llvm['loop_blocks']}: store i64 = {llvm['store_i64_in_loop']}, "
                  f"load i64 = {llvm['load_i64_in_loop']}, vector ops = {llvm['vector_ops_in_loop']}, "
                  f"calls = {llvm['native_calls']}, pyobject = {llvm['pyobject_fallback']}")
        for l in llvm["loop_excerpt"]:
            log0(ctx, f"      {l}")
        log0(ctx, f"  llvm_audit_passed: {llvm['passed']}")

        shm_exact = all(v["counters_bit_exact"] for v in shm.values())
        verification = {
            "mpi_ranks_2x2": ctx["size"] == 4 and ctx["dims"] == [2, 2],
            "gpu_parity_below_1e-12": parity_max_diff < 1e-12,
            "energy_monotonic": energy_monotonic,
            "ptx_audit_passed": ptx_ok,
            "llvm_audit_passed": llvm["passed"],
            "shm_counters_bit_exact": shm_exact,
            "shm_window_64B_aligned": ctx["shm_base_mod64"] == 0,
        }
        verification_passed = all(verification.values())
        with open(os.path.abspath(__file__), "rb") as fh:
            script_sha = hashlib.sha256(fh.read()).hexdigest()
        import numba
        import mpi4py
        manifest = {
            "github_username": ctx["github_user"], "gpu_name": ctx["gpu_name"], "gpu_uuid": ctx["gpu_uuid"],
            "cpu_model": ctx["cpu_model"], "cpu_sig": ctx["cpu_sig"], "cpu_line_bytes": ctx["cpu_line_bytes"],
            "instance_seed": ctx["instance_seed"], "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
            "mpi_ranks": ctx["size"], "cart_dims": ctx["dims"], "cart_periods": ctx["periods"],
            "node_comm_size": ctx["node_size"],
            "rank_topology": [{"rank": r, "coords": [int(table[r, 0]), int(table[r, 1])],
                               "up": int(table[r, 2]), "down": int(table[r, 3]), "left": int(table[r, 4]),
                               "right": int(table[r, 5]), "local_shape": [int(table[r, 6]), int(table[r, 7])]}
                              for r in range(ctx["size"])],
            "rank_cpu_affinity": sorted(aff),
            "global_grid": [GLOBAL_NY, GLOBAL_NX], "local_padded_shape": [ny + 2, nx + 2],
            "tile_dim": TILE_DIM, "shared_dim": SHARED_DIM, "stencil_steps": STENCIL_STEPS,
            "lambda_damp": LAMBDA_DAMP, "dx": dx, "kappa_min": kappa_min, "kappa_max": kappa_max,
            "dt_max": dt_max, "dt": dt, "cfl_safety": CFL_SAFETY, "source_integral": src_integral,
            "seeded_parameters": params,
            "energy_trace_serialized": gpu["serialized"]["trace"],
            "energy_trace_overlapped": gpu["overlapped"]["trace"],
            "energy_monotonic_non_increasing": energy_monotonic,
            "parity_max_diff": parity_max_diff,
            "gpu_serialized_ms": gpu_serialized_ms, "gpu_overlapped_ms": gpu_overlapped_ms,
            "gpu_overlap_speedup": gpu_serialized_ms / gpu_overlapped_ms,
            "cuda_event_timings_ms": cuda_events,
            "ptx_counts": ptx_counts, "ptx_audit_passed": ptx_ok,
            "shm_window": {"slots": SHM_SLOTS, "base_mod64": ctx["shm_base_mod64"],
                           "base_mod128": ctx["shm_base_mod128"], "disp_unit": ctx["shm_disp_unit"]},
            "shm_iters": SHM_ITERS, "shm_trials": SHM_TRIALS, "shm_results": shm,
            "shm_false_sharing_slowdown_stride1_vs_stride8": shm_slowdown,
            "shm_ratio_stride16_vs_stride8": shm["stride_16"]["median_ns"] / s8,
            "llvm_audit": llvm, "llvm_audit_passed": llvm["passed"],
            "verification": verification, "verification_passed": verification_passed,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "numba": numba.__version__,
                         "mpi4py": mpi4py.__version__,
                         "mpi_library": MPI.Get_library_version().splitlines()[0].strip()},
            "script_sha256": script_sha,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        log0(ctx, "\n[Task 5] Verification")
        for k, v in verification.items():
            log0(ctx, f"  {k:26s} {'OK' if v else 'FAIL'}")
        log0(ctx, f"  verification_passed: {verification_passed}")
        log0(ctx, f"  {MANIFEST_PATH} written (sha256 {manifest['manifest_sha256'][:16]}...)")

    ctx["cart_comm"].Barrier()
    ctx["win"].Unlock_all()
    ctx["win"].Free()


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("usage: mpirun -np 4 python3 hybrid_mpi_cuda_cache.py <github_username>")
    main(sys.argv[1].strip())
