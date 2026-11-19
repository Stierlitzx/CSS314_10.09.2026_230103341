# Hybrid MPI-CUDA Distributed Stencil & Cross-Tier Cache-Line Coherency (Python)

### Practice lab (ungraded, solo) · 19 November 2026 · 14:30

**Ayazgaliyev Aibek · Student ID `230103341` · GitHub `Stierlitzx`** · Instructor: Sufyan bin Uzayr

Four MPI ranks run a non-periodic 2 × 2 Cartesian MPI + CUDA reaction-diffusion stencil on a 512 × 512 grid (Tiers
1–3: 258 × 258 rank-local arrays, GPU halo pack/unpack, pinned `Isend`/`Irecv`, 18 × 18 shared-memory tile, dual-stream
overlap). The same ranks then run an MPI-3 `Win.Allocate_shared` false-sharing sweep with a `@njit(nogil=True)` kernel
(Tier 4). Both a GPU PTX audit and a CPU LLVM IR audit are performed.

**Executed on Google Colab: Tesla T4 `GPU-6c19b838-7a5d-1355-0fdd-83ef8e21deb2`, Intel Xeon @ 2.00 GHz with 2 vCPUs
(one physical core, 2 hyper-threads), Open MPI 4.1.6.** `INSTANCE_SEED = 8070547215371552121` (`0x700057eecb3d4579`).

## Results at a glance

| Check | Result |
|---|---|
| 4 ranks, 2 × 2 grid, local shape `(258, 258)` | **OK** (rank 0 `(0,0)`: up/left = `PROC_NULL`) |
| GPU parity serialized vs overlapped (300 steps) | **0.0** (< 1e-12) |
| PTX audit `tiled_stencil_kernel` | **`st.shared.f64` 5 · `ld.shared.f64` 5 · `bar.sync` 1** |
| LLVM IR audit `host_shm_accumulate` | loop `B34`: **5 `load i64` + 5 `store i64`**, no vector ops, no calls, no Python objects |
| Shared window | 64 `uint64` slots, base **128-B aligned** (`% 64 = 0`, `% 128 = 0`) |
| Shared-window counters (3 strides × 3 trials × 4 ranks) | **bit-exact** vs register reference |
| Global energy (allreduce), steps 0 / 100 / 200 / 300 | 5.7979089e-2 → 5.7975669e-2 → 5.7972411e-2 → 5.7969289e-2 (**monotone ↓**) |
| **`verification_passed`** | **true** |

| Timing | Value |
|---|---:|
| `gpu_serialized_ms` (300 steps) | 1,182.134 ms |
| `gpu_overlapped_ms` (300 steps) | 1,411.503 ms (0.838× — host oversubscribed: 4 ranks on 2 vCPUs, see Q3) |
| Interior kernel (CUDA event) | 315.29 µs/step |
| Shared window `stride 1` (8 B, 1 line) | **579,522,133 ns** (14.488 ns/update) |
| Shared window `stride 8` (64 B) | 192,220,911 ns (4.806 ns/update) |
| Shared window `stride 16` (128 B) | 193,755,158 ns (4.844 ns/update) |
| **False-sharing slowdown stride 1 / stride 8** | **3.015×** on Colab · **6.317×** on the 6-core Ryzen cross-check |
| CFL | `Δt_max = 3.958233e-6`, `Δt = 3.562410e-6` (`κ_max·Δt/Δx² = 0.225`) |

## Files

| File | What it is |
|---|---|
| [`hybrid_mpi_cuda_cache_lab.ipynb`](hybrid_mpi_cuda_cache_lab.ipynb) | Notebook executed on the Colab T4 and downloaded from Colab (all outputs), plus the Q1–Q6 answers cell |
| [`hybrid_mpi_cuda_cache.py`](hybrid_mpi_cuda_cache.py) | Engine, identical to the `%%writefile` cell (manifest `script_sha256` matches) |
| [`hybrid_lab_manifest.json`](hybrid_lab_manifest.json) | Manifest written on Colab, taken verbatim from the notebook output; `manifest_sha256` recomputes |
| [`hybrid_stdout_colab_t4.txt`](hybrid_stdout_colab_t4.txt) | Raw `mpirun -np 4` stdout of the Colab run |
| [`local_ryzen_shm_crosscheck.txt`](local_ryzen_shm_crosscheck.txt) | Local cross-check on a 6-core Ryzen with MS-MPI. Only the Tier-4/LLVM numbers are real; the GPU tiers ran in the CUDA simulator on a 40 × 40 grid, also checked against a NumPy reference (≤ 1.1e-16) |
| [`answers_q1_q6.md`](answers_q1_q6.md) | Source of the Q1–Q6 cell |
| [`make_notebook.py`](make_notebook.py) | Builds the notebook from the engine file; `--answers` refreshes only the Q1–Q6 cell |

## Implementation map

| Task | Function(s) |
|---|---|
| 1 — hybrid context | `init_hybrid_context`: `Create_cart([2,2], [False,False])`, `Shift` neighbours, `cuda.select_device(0)`, `Split_type(COMM_TYPE_SHARED)`, `Win.Allocate_shared` + `Shared_query` with a 128-B aligned view, `Lock_all`; seed = SHA-256(`user::gpu_uuid::cpu_sig`) |
| 2 — fields & halos | `initialize_fields` (analytic global coordinates, no global grid), `pack_halos_kernel`, `unpack_halos_kernel` |
| 3 — tiled stencil & overlap | `tiled_stencil_kernel` (16 × 16 threads, 18 × 18 `cuda.shared.array`, `syncthreads`), `step_overlapped` (interior `[2..255]²` on `stream_interior`, pinned exchange + 4 border strips on `stream_boundary`), `step_serialized` |
| 4 — host false sharing | `host_shm_accumulate` (`@njit(nogil=True)`, address `slot + (x & mask)` with run-time `mask = 0` prevents register hoisting), `benchmark_shm_strides` (`node_comm.Barrier` + `win.Sync`, strides 1/8/16, 3 trials, slowest rank) |
| 5 — verification | `main`: parity, `cart_comm.allreduce` energies, PTX audit (`inspect_asm`), LLVM audit (`inspect_llvm`), `hybrid_lab_manifest.json` |

## Run history (honest record)

The first Colab run on another T4 VM failed one check (`shm_counters_bit_exact`). This was a race in my benchmark: after
the closing barrier, node rank 0 zeroed the window for the next trial while slower ranks were still reading their
counters (4 ranks on 2 vCPUs exposed it; it never appeared on 12 local threads). The fix is for each rank to read its
own slot immediately after its own kernel; no other rank writes it. The notebook was rebuilt, re-uploaded and
re-executed from the top on a fresh T4 runtime, and that run is the one committed here.

> Submission protocol note: the assignment asks for a repository named `hybrid-mpi-cuda-cache-Stierlitzx` with
> `hybrid_mpi_cuda_cache_lab.ipynb`, `hybrid_mpi_cuda_cache.py` and `hybrid_lab_manifest.json` at its root. This folder
> holds those three files.
