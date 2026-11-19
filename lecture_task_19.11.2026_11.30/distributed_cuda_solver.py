import os
import sys
import json
import math
import hashlib
import subprocess
import time
import re
import warnings
import numpy as np

# Small border-ring launches (1x16 / 16x1 blocks) are intentional; silence Numba's occupancy hint.
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15

# Additional solver constants (not part of the instructor schema, recorded in the manifest)
DIAG_EVERY = 100
CFL_SAFETY = 0.90          # dt = CFL_SAFETY * dx^2 / (4 * kappa_max)
WARMUP_STEPS = 3           # JIT-compile + warm caches before timed runs
PACK_THREADS = 128

# Message tags = direction of travel of the payload
TAG_TO_UP, TAG_TO_DOWN, TAG_TO_LEFT, TAG_TO_RIGHT = 10, 11, 12, 13
# Row index of each strip inside the (4, n) send / receive buffers
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def get_gpu_name() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def log0(ctx, *args):
    if ctx["rank"] == 0:
        print(*args, flush=True)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    gpu_name = get_gpu_name() if rank == 0 else None
    gpu_name = comm.bcast(gpu_name, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    dims = [2, 2]
    periods = [False, False]
    cart_comm = comm.Create_cart(dims=dims, periods=periods, reorder=False)
    coords = cart_comm.Get_coords(cart_comm.Get_rank())
    nbr_up, nbr_down = cart_comm.Shift(0, 1)      # axis 0: rows (y direction)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)   # axis 1: columns (x direction)

    ny = GLOBAL_NY // dims[0]
    nx = GLOBAL_NX // dims[1]
    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": cart_comm.Get_rank(),
        "size": size,
        "dims": dims,
        "periods": periods,
        "coords": tuple(int(c) for c in coords),
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "ny": ny,
        "nx": nx,
        "github_user": github_user,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "instance_seed": instance_seed,
    }


def rank_table(ctx):
    """Collective per-rank metadata table built with Allreduce (no field arrays are collected)."""
    cart = ctx["cart_comm"]
    pn = MPI.PROC_NULL
    row = np.zeros((ctx["size"], 9), dtype=np.float64)
    nbrs = [ctx["nbr_up"], ctx["nbr_down"], ctx["nbr_left"], ctx["nbr_right"]]
    row[ctx["rank"]] = [
        ctx["coords"][0], ctx["coords"][1],
        *[(-1 if n == pn else n) for n in nbrs],
        ctx["ny"] + 2, ctx["nx"] + 2, os.getpid(),
    ]
    table = np.zeros_like(row)
    cart.Allreduce(row, table, op=MPI.SUM)
    return table


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ==============================================================================
def seeded_parameters(instance_seed: int):
    """All ranks derive identical physical parameters from the same 64-bit seed."""
    rng = np.random.Generator(np.random.PCG64(instance_seed))
    n_pulses = 3
    params = {
        "pulse_cx": rng.uniform(0.22, 0.78, n_pulses).tolist(),
        "pulse_cy": rng.uniform(0.22, 0.78, n_pulses).tolist(),
        "pulse_amp": rng.uniform(0.60, 1.40, n_pulses).tolist(),
        "pulse_sigma": rng.uniform(0.040, 0.070, n_pulses).tolist(),
        "kappa_fx": int(rng.integers(1, 4)),
        "kappa_fy": int(rng.integers(1, 4)),
        "kappa_phx": float(rng.uniform(0.0, 1.0)),
        "kappa_phy": float(rng.uniform(0.0, 1.0)),
        # Localised source/sink dipole (equal and opposite lobes -> zero net injection)
        "src_amp": float(rng.uniform(3.0, 6.0)),
        "src_sigma": 0.030,
        "src_pos": [float(v) for v in rng.uniform(0.30, 0.70, 2)],
        "sink_pos": [float(v) for v in rng.uniform(0.30, 0.70, 2)],
    }
    return params


def initialize_subdomains(ctx, params):
    """Build rank-local (ny+2, nx+2) host fields from analytic global coordinates.

    No global 512x512 array is ever allocated. Local index li in 0..ny+1 maps to global
    index gi = row_rank * ny + li, with physical coordinate y = gi * dy (dy = 1/513), so
    gi = 0 and gi = 513 are the Dirichlet walls and 1..512 are the interior cells.
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
    src = p["src_amp"] * (
        np.exp(-((X - p["src_pos"][0]) ** 2 + (Y - p["src_pos"][1]) ** 2) / s2)
        - np.exp(-((X - p["sink_pos"][0]) ** 2 + (Y - p["sink_pos"][1]) ** 2) / s2)
    )

    # Dirichlet u = 0 on outer global edges (neighbour == MPI.PROC_NULL)
    if ctx["nbr_up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return {
        "u": np.ascontiguousarray(u),
        "kappa": np.ascontiguousarray(kappa),
        "src": np.ascontiguousarray(src),
        "dx": dx,
        "dy": dy,
    }


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, send_buf, ny, nx):
    """Gather the four owned border strips (two of them strided columns) into contiguous rows."""
    k = cuda.grid(1)
    if k < nx:
        send_buf[0, k] = u[1, k + 1]       # top owned row     -> neighbour up
        send_buf[1, k] = u[ny, k + 1]      # bottom owned row  -> neighbour down
    if k < ny:
        send_buf[2, k] = u[k + 1, 1]       # left owned column  (stride nx+2) -> neighbour left
        send_buf[3, k] = u[k + 1, nx]      # right owned column (stride nx+2) -> neighbour right


@cuda.jit
def unpack_halos_kernel(u, recv_buf, ny, nx, has_up, has_down, has_left, has_right):
    """Scatter received contiguous vectors into ghost rows/columns 0 and n+1."""
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
    bufs = {
        "d_send": cuda.device_array((4, n), dtype=np.float64),
        "d_recv": cuda.to_device(np.zeros((4, n), dtype=np.float64)),
        "h_send": cuda.pinned_array((4, n), dtype=np.float64),
        "h_recv": cuda.pinned_array((4, n), dtype=np.float64),
    }
    bufs["h_send"][:] = 0.0
    bufs["h_recv"][:] = 0.0
    return bufs


def exchange_halos_async(ctx, bufs, d_u, stream):
    """Stage A + Stage B (posting): GPU pack -> async D2H into pinned memory -> Irecv/Isend.

    Returns the list of outstanding MPI requests; the caller overlaps work and then calls
    complete_halo_exchange().
    """
    ny, nx = ctx["ny"], ctx["nx"]
    cart = ctx["cart_comm"]
    n = bufs["d_send"].shape[1]
    blocks = (n + PACK_THREADS - 1) // PACK_THREADS
    pack_halos_kernel[blocks, PACK_THREADS, stream](d_u, bufs["d_send"], ny, nx)
    bufs["d_send"].copy_to_host(bufs["h_send"], stream=stream)
    stream.synchronize()  # pinned send buffers are now valid on the host

    hs, hr = bufs["h_send"], bufs["h_recv"]
    reqs = [
        # receive what neighbours sent towards me
        cart.Irecv([hr[UP, :nx], MPI.DOUBLE], source=ctx["nbr_up"], tag=TAG_TO_DOWN),
        cart.Irecv([hr[DOWN, :nx], MPI.DOUBLE], source=ctx["nbr_down"], tag=TAG_TO_UP),
        cart.Irecv([hr[LEFT, :ny], MPI.DOUBLE], source=ctx["nbr_left"], tag=TAG_TO_RIGHT),
        cart.Irecv([hr[RIGHT, :ny], MPI.DOUBLE], source=ctx["nbr_right"], tag=TAG_TO_LEFT),
        # send my owned border strips
        cart.Isend([hs[UP, :nx], MPI.DOUBLE], dest=ctx["nbr_up"], tag=TAG_TO_UP),
        cart.Isend([hs[DOWN, :nx], MPI.DOUBLE], dest=ctx["nbr_down"], tag=TAG_TO_DOWN),
        cart.Isend([hs[LEFT, :ny], MPI.DOUBLE], dest=ctx["nbr_left"], tag=TAG_TO_LEFT),
        cart.Isend([hs[RIGHT, :ny], MPI.DOUBLE], dest=ctx["nbr_right"], tag=TAG_TO_RIGHT),
    ]
    return reqs


def complete_halo_exchange(ctx, bufs, d_u, reqs, stream):
    """Stage C (part 1): Waitall -> async H2D of pinned halos -> GPU unpack into ghosts."""
    MPI.Request.Waitall(reqs)
    bufs["d_recv"].copy_to_device(bufs["h_recv"], stream=stream)
    n = bufs["d_recv"].shape[1]
    blocks = (n + PACK_THREADS - 1) // PACK_THREADS
    pn = MPI.PROC_NULL
    unpack_halos_kernel[blocks, PACK_THREADS, stream](
        d_u, bufs["d_recv"], ctx["ny"], ctx["nx"],
        int(ctx["nbr_up"] != pn), int(ctx["nbr_down"] != pn),
        int(ctx["nbr_left"] != pn), int(ctx["nbr_right"] != pn),
    )


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, dt, inv_dx2, inv_dy2, lam,
                         r_start, r_end, c_start, c_end):
    """Update u_new[r_start..r_end, c_start..c_end] (inclusive) with an 18x18 shared tile.

    Reads are confined to the cross-shaped neighbourhood of the box, so a launch on the
    deep interior [2..n-1]^2 never touches ghost cells 0 / n+1.
    """
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty   # global row in padded array
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx   # global column in padded array
    sy = ty + 1
    sx = tx + 1

    # Centre cell (also serves as apron for a neighbour when the tile is partial)
    if i <= r_end + 1 and j <= c_end + 1:
        smem[sy, sx] = u[i, j]
    # 1-cell halo apron loaded by the tile's edge threads
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
    rows = r1 - r0 + 1
    cols = c1 - c0 + 1
    grid = ((cols + TILE_DIM - 1) // TILE_DIM, (rows + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        state["d_u"], state["d_u_new"], state["d_kappa"], state["d_src"],
        state["dt"], state["inv_dx2"], state["inv_dy2"], LAMBDA_DAMP,
        r0, r1, c0, c1,
    )


def step_serialized(ctx, state, bufs, streams, timers):
    """Reference path: exchange fully, then one full-domain stencil on a single stream."""
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
    """Dual-stream pipeline: Stage A/B on stream_boundary, deep interior on stream_interior."""
    sb, si = streams["boundary"], streams["interior"]
    ny, nx = ctx["ny"], ctx["nx"]
    ev = timers["ev"]

    ev[0].record(sb)
    reqs = exchange_halos_async(ctx, bufs, state["d_u"], sb)           # Stage A (+ post B)
    ev[1].record(sb)

    ev[2].record(si)
    launch_stencil(state, si, 2, ny - 1, 2, nx - 1)                     # Stage B: deep interior
    ev[3].record(si)

    ev[4].record(sb)
    complete_halo_exchange(ctx, bufs, state["d_u"], reqs, sb)          # Stage C: wait/H2D/unpack
    launch_stencil(state, sb, 1, 1, 1, nx)                              # top ring row
    launch_stencil(state, sb, ny, ny, 1, nx)                            # bottom ring row
    launch_stencil(state, sb, 2, ny - 1, 1, 1)                          # left ring column
    launch_stencil(state, sb, 2, ny - 1, nx, nx)                        # right ring column
    ev[5].record(sb)

    sb.synchronize()
    si.synchronize()
    timers["pack_ms"] += cuda.event_elapsed_time(ev[0], ev[1])
    timers["interior_ms"] += cuda.event_elapsed_time(ev[2], ev[3])
    timers["boundary_ms"] += cuda.event_elapsed_time(ev[4], ev[5])
    state["d_u"], state["d_u_new"] = state["d_u_new"], state["d_u"]


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def make_state(ctx, fields, dt):
    return {
        "d_u": cuda.to_device(fields["u"]),
        "d_u_new": cuda.to_device(fields["u"]),   # carries identical Dirichlet ghosts
        "d_kappa": cuda.to_device(fields["kappa"]),
        "d_src": cuda.to_device(fields["src"]),
        "dt": float(dt),
        "inv_dx2": 1.0 / fields["dx"] ** 2,
        "inv_dy2": 1.0 / fields["dy"] ** 2,
    }


def global_diagnostics(ctx, state, fields, step):
    """Energy E = sum(u dx dy), L2 residual ||u^n - u^(n-1)||_2, max|u| via Allreduce."""
    ny, nx = ctx["ny"], ctx["nx"]
    u = state["d_u"].copy_to_host()[1:ny + 1, 1:nx + 1]
    u_prev = state["d_u_new"].copy_to_host()[1:ny + 1, 1:nx + 1]
    local = np.array([
        u.sum() * fields["dx"] * fields["dy"],
        0.0 if step == 0 else float(((u - u_prev) ** 2).sum()),
    ])
    glob = np.zeros(2)
    ctx["cart_comm"].Allreduce(local, glob, op=MPI.SUM)
    umax_l = np.array([np.abs(u).max()])
    umax_g = np.zeros(1)
    ctx["cart_comm"].Allreduce(umax_l, umax_g, op=MPI.MAX)
    return {"step": step, "energy": float(glob[0]), "l2_residual": float(math.sqrt(glob[1])),
            "u_max": float(umax_g[0])}


def run_mode(ctx, fields, dt, bufs, streams, mode, steps, record=True):
    state = make_state(ctx, fields, dt)
    timers = {"ev": [cuda.event(timing=True) for _ in range(6)],
              "full_ms": 0.0, "pack_ms": 0.0, "interior_ms": 0.0, "boundary_ms": 0.0}
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    trace = []
    if record:
        trace.append(global_diagnostics(ctx, state, fields, 0))
    ctx["cart_comm"].Barrier()
    t_start = MPI.Wtime()
    for n in range(1, steps + 1):
        step_fn(ctx, state, bufs, streams, timers)
        if record and n % DIAG_EVERY == 0:
            trace.append(global_diagnostics(ctx, state, fields, n))
    ctx["cart_comm"].Barrier()
    wall_ms = (MPI.Wtime() - t_start) * 1e3
    timers.pop("ev")
    return state, trace, wall_ms, timers


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", ptx)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", ptx)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", ptx)),
        "barrier.sync": len(re.findall(r"\bbarrier\.sync\b", ptx)),
        "st.shared (any)": len(re.findall(r"\bst\.shared\.", ptx)),
        "ld.shared (any)": len(re.findall(r"\bld\.shared\.", ptx)),
    }
    passed = (counts["st.shared (any)"] > 0 and counts["ld.shared (any)"] > 0
              and (counts["bar.sync"] + counts["barrier.sync"]) > 0)
    return ptx, counts, passed


def reduce_scalar(ctx, value, op):
    out = np.zeros(1)
    ctx["cart_comm"].Allreduce(np.array([float(value)]), out, op=op)
    return float(out[0])


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    rank, ny, nx = ctx["rank"], ctx["ny"], ctx["nx"]

    log0(ctx, "=" * 78)
    log0(ctx, "Distributed MPI + CUDA anisotropic reaction-diffusion solver")
    log0(ctx, "=" * 78)
    log0(ctx, f"GitHub identity      : {ctx['github_user']}")
    log0(ctx, f"GPU                  : {ctx['gpu_name']}")
    log0(ctx, f"GPU UUID             : {ctx['gpu_uuid']}")
    log0(ctx, f"INSTANCE_SEED        : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    log0(ctx, f"MPI library          : {MPI.Get_library_version().splitlines()[0].strip()}")
    log0(ctx, f"MPI ranks            : {ctx['size']} in a {ctx['dims'][0]}x{ctx['dims'][1]} "
              f"Cartesian grid, periods={ctx['periods']}")

    table = rank_table(ctx)
    log0(ctx, "\n[Task 1] Cartesian topology (PROC_NULL shown as '-'):")
    log0(ctx, "  rank  coords   up  down  left  right   local_shape   pid")
    for r in range(ctx["size"]):
        t = table[r].astype(int)
        f = lambda v: " -" if v < 0 else f"{v:2d}"
        log0(ctx, f"  {r:4d}  ({t[0]},{t[1]})   {f(t[2])}   {f(t[3])}    {f(t[4])}    {f(t[5])}"
                  f"    ({t[6]}, {t[7]})   {t[8]}")

    # ---- Task 2
    params = seeded_parameters(ctx["instance_seed"])
    fields = initialize_subdomains(ctx, params)
    kappa_max = reduce_scalar(ctx, fields["kappa"].max(), MPI.MAX)
    kappa_min = reduce_scalar(ctx, fields["kappa"].min(), MPI.MIN)
    owned = (slice(1, ny + 1), slice(1, nx + 1))
    src_integral = reduce_scalar(ctx, fields["src"][owned].sum() * fields["dx"] * fields["dy"], MPI.SUM)
    dx = fields["dx"]
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max
    log0(ctx, f"\n[Task 2] Seeded subdomain initialisation (rank-local {ny + 2}x{nx + 2} arrays only)")
    log0(ctx, f"  dx = dy = 1/{GLOBAL_NX + 1} = {dx:.10e}")
    log0(ctx, f"  kappa range (global Allreduce MIN/MAX) = [{kappa_min:.6f}, {kappa_max:.6f}]")
    log0(ctx, f"  pulses: centres x={np.round(params['pulse_cx'], 4).tolist()} "
              f"y={np.round(params['pulse_cy'], 4).tolist()}")
    log0(ctx, f"          amplitudes={np.round(params['pulse_amp'], 4).tolist()} "
              f"sigmas={np.round(params['pulse_sigma'], 4).tolist()}")
    log0(ctx, f"  source dipole amp={params['src_amp']:.4f} src@{np.round(params['src_pos'], 4).tolist()} "
              f"sink@{np.round(params['sink_pos'], 4).tolist()}  integral S dA = {src_integral:.3e}")
    log0(ctx, f"  CFL: dt_max = dx^2/(4 kappa_max) = {dt_max:.6e}; dt = {CFL_SAFETY} * dt_max = {dt:.6e}")

    # ---- Task 3/4 resources
    bufs = allocate_exchange_buffers(ctx)
    streams = {"boundary": cuda.stream(), "interior": cuda.stream()}

    # Warm-up (JIT compilation) for both paths, not timed
    run_mode(ctx, fields, dt, bufs, streams, "serialized", WARMUP_STEPS, record=False)
    run_mode(ctx, fields, dt, bufs, streams, "overlapped", WARMUP_STEPS, record=False)

    log0(ctx, f"\n[Task 5] Running {STEPS} steps per mode (diagnostics every {DIAG_EVERY} steps)")
    results = {}
    for mode in ("serialized", "overlapped"):
        state, trace, wall_ms, timers = run_mode(ctx, fields, dt, bufs, streams, mode, STEPS)
        results[mode] = {"state": state, "trace": trace, "wall_ms": wall_ms, "timers": timers}
        log0(ctx, f"\n  -- {mode} mode --")
        log0(ctx, "     step        energy E (Allreduce)        L2 residual           max|u|")
        for d in trace:
            log0(ctx, f"     {d['step']:4d}   {d['energy']:.15e}   {d['l2_residual']:.12e}   {d['u_max']:.10f}")

    # Parity between serialized and overlapped final states (rank-local, then Allreduce MAX)
    u_ser = results["serialized"]["state"]["d_u"].copy_to_host()
    u_ovl = results["overlapped"]["state"]["d_u"].copy_to_host()
    parity_max_diff = reduce_scalar(ctx, np.abs(u_ser[owned] - u_ovl[owned]).max(), MPI.MAX)
    checksum = reduce_scalar(ctx, float((u_ovl[owned] ** 2).sum()), MPI.SUM)

    # Timing: slowest rank defines wall time; CUDA event times averaged and maxed over ranks
    serialized_ms = reduce_scalar(ctx, results["serialized"]["wall_ms"], MPI.MAX)
    overlapped_ms = reduce_scalar(ctx, results["overlapped"]["wall_ms"], MPI.MAX)
    ev_ser = results["serialized"]["timers"]
    ev_ovl = results["overlapped"]["timers"]
    cuda_events = {}
    for name, val in [("serialized_step_total_ms", ev_ser["full_ms"]),
                      ("overlapped_stageA_pack_d2h_ms", ev_ovl["pack_ms"]),
                      ("overlapped_interior_kernel_ms", ev_ovl["interior_ms"]),
                      ("overlapped_stageC_boundary_ms", ev_ovl["boundary_ms"])]:
        cuda_events[name] = {
            "mean_over_ranks": reduce_scalar(ctx, val, MPI.SUM) / ctx["size"],
            "max_over_ranks": reduce_scalar(ctx, val, MPI.MAX),
            "per_step_mean_ms": reduce_scalar(ctx, val, MPI.SUM) / ctx["size"] / STEPS,
        }

    log0(ctx, "\n[Task 5] Parity & benchmark")
    log0(ctx, f"  parity_max_diff (serialized vs overlapped) = {parity_max_diff:.3e}  "
              f"-> {'PASS' if parity_max_diff < 1e-12 else 'FAIL'} (< 1e-12)")
    log0(ctx, f"  global sum(u^2) checksum                   = {checksum:.15e}")
    log0(ctx, f"  serialized_ms = {serialized_ms:.3f} ms  ({serialized_ms / STEPS:.4f} ms/step)")
    log0(ctx, f"  overlapped_ms = {overlapped_ms:.3f} ms  ({overlapped_ms / STEPS:.4f} ms/step)")
    log0(ctx, f"  overlap speed-up = {serialized_ms / overlapped_ms:.4f}x")
    log0(ctx, f"  CUDA event timings (sum over {STEPS} steps, mean over ranks):")
    for k, v in cuda_events.items():
        log0(ctx, f"    {k:34s} {v['mean_over_ranks']:10.3f} ms  (max rank {v['max_over_ranks']:.3f} ms, "
                  f"{v['per_step_mean_ms'] * 1e3:.2f} us/step)")

    # PTX audit (rank 0 inspects the compiled kernel)
    if rank == 0:
        ptx, counts, ptx_passed = ptx_audit()
        with open("tiled_stencil_kernel.ptx", "w", encoding="utf-8") as fh:
            fh.write(ptx)
        log0(ctx, "\n[Task 5] PTX audit of tiled_stencil_kernel.inspect_asm()")
        for k, v in counts.items():
            log0(ctx, f"  {k:18s} : {v}")
        log0(ctx, f"  ptx_audit_passed   : {ptx_passed}")

        trace_ser = results["serialized"]["trace"]
        trace_ovl = results["overlapped"]["trace"]
        energies = [d["energy"] for d in trace_ovl]
        monotonic = all(b <= a for a, b in zip(energies, energies[1:]))
        with open(os.path.abspath(__file__), "rb") as fh:
            solver_sha = hashlib.sha256(fh.read()).hexdigest()
        import numba
        import mpi4py
        manifest = {
            "github_username": ctx["github_user"],
            "gpu_name": ctx["gpu_name"],
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["instance_seed"],
            "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
            "mpi_ranks": ctx["size"],
            "cart_dims": ctx["dims"],
            "cart_periods": ctx["periods"],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [ny + 2, nx + 2],
            "tile_dim": TILE_DIM,
            "shared_dim": SHARED_DIM,
            "steps": STEPS,
            "lambda_damp": LAMBDA_DAMP,
            "dx": dx,
            "kappa_min": kappa_min,
            "kappa_max": kappa_max,
            "dt_max": dt_max,
            "dt": dt,
            "cfl_safety": CFL_SAFETY,
            "source_integral": src_integral,
            "seeded_parameters": params,
            "energy_trace_serialized": trace_ser,
            "energy_trace_overlapped": trace_ovl,
            "energy_monotonic_non_increasing": monotonic,
            "parity_max_diff": parity_max_diff,
            "parity_passed": parity_max_diff < 1e-12,
            "final_sum_u2_checksum": checksum,
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "overlap_speedup": serialized_ms / overlapped_ms,
            "cuda_event_timings_ms": cuda_events,
            "ptx_counts": counts,
            "ptx_audit_passed": ptx_passed,
            "rank_topology": [
                {"rank": r, "coords": [int(table[r, 0]), int(table[r, 1])],
                 "up": int(table[r, 2]), "down": int(table[r, 3]),
                 "left": int(table[r, 4]), "right": int(table[r, 5]),
                 "local_shape": [int(table[r, 6]), int(table[r, 7])]}
                for r in range(ctx["size"])
            ],
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "numba": numba.__version__,
                "mpi4py": mpi4py.__version__,
                "mpi_library": MPI.Get_library_version().splitlines()[0].strip(),
            },
            "solver_sha256": solver_sha,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        body = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(body).hexdigest()
        with open("submission_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        log0(ctx, f"\n  energy monotonic non-increasing (overlapped) : {monotonic}")
        log0(ctx, f"  submission_manifest.json written (sha256 {manifest['manifest_sha256'][:16]}...)")
        all_ok = ptx_passed and parity_max_diff < 1e-12 and ctx["size"] == 4
        log0(ctx, f"\nALL TASKS COMPLETE: {'YES' if all_ok else 'NO'}")
    ctx["cart_comm"].Barrier()


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    run_solver(sys.argv[1].strip())
