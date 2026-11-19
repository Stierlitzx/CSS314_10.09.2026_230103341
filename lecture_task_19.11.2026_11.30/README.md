# Distributed Multi-Rank Stencil & Asynchronous Halo Exchange in MPI + CUDA (Python)

### Laboratory deliverable · 19 November 2026 · 11:30

**Ayazgaliyev Aibek · Student ID `230103341` · GitHub `Stierlitzx`** · Instructor: Sufyan bin Uzayr

This is a 4-rank anisotropic reaction-diffusion solver on a 512 × 512 grid, written with `mpi4py` + `numba.cuda`. Each
rank owns a 258 × 258 padded device array in a non-periodic 2 × 2 Cartesian grid. Ranks exchange halos through GPU
pack/unpack kernels and page-locked host buffers with `Isend`/`Irecv`. Cells are updated by an 18 × 18 shared-memory
tiled stencil on two CUDA streams: the deep interior on `stream_interior`, and the boundary ring on `stream_boundary`.

**Executed on Google Colab — Tesla T4 (CC 7.5), UUID `GPU-198f4157-3506-5952-ab80-80b6dd077a59`, Open MPI 4.1.6.**

## Results at a glance

| Verification item | Required | This run |
|---|---|---|
| `GITHUB_USERNAME` | real handle | **`Stierlitzx`** |
| `INSTANCE_SEED` | SHA-256(user ‖ UUID) | `15753744789197117896` (`0xdaa08b6b3a4159c8`) |
| `mpi_ranks` | 4 in 2 × 2 | **4**, dims `[2, 2]`, periods `[False, False]` |
| `ptx_audit_passed` | non-zero shared/barrier counts | **true** — `st.shared.f64` 5, `ld.shared.f64` 5, `bar.sync` 1 |
| `parity_max_diff` | < 1e-12 | **0.0** (bit-identical) |
| Energy trace | dissipative | **monotone ↓** 7.6067621e-2 → 7.6060523e-2 |

| Benchmark (400 steps, slowest rank) | Wall time | Per step |
|---|---:|---:|
| `serialized_ms` | 3,869.639 ms | 9.674 ms |
| `overlapped_ms` | **1,627.154 ms** | **4.068 ms** |
| Overlap speed-up | **2.378×** | |

| CUDA event (mean over ranks) | Per step |
|---|---:|
| Stage A: pack + async D2H + sync | 209.56 µs |
| Interior 254 × 254 tiled kernel (`stream_interior`) | 298.33 µs |
| Stage C: Waitall + H2D + unpack + 4 ring kernels | 3,191.40 µs |
| Serialized step (exchange + full stencil) | 9,338.05 µs |

The CFL limit is `Δt_max = Δx²/(4κ_max) = 3.958193e-6` with `κ_max = 0.2399984`. The run uses `Δt = 0.9·Δt_max = 3.562373e-6`.

## Files

| File | What it is |
|---|---|
| [`mpi_cuda_stencil_lab.ipynb`](mpi_cuda_stencil_lab.ipynb) | The notebook as executed on the Colab T4 and downloaded from Colab (all cell outputs), plus the Q1–Q6 Markdown answers cell |
| [`distributed_cuda_solver.py`](distributed_cuda_solver.py) | The solver; identical to the notebook's `%%writefile` cell (the manifest's `solver_sha256` matches this file) |
| [`submission_manifest.json`](submission_manifest.json) | Manifest written by rank 0 on Colab, taken verbatim from the notebook output; `manifest_sha256` recomputes correctly |
| [`solver_stdout_colab_t4.txt`](solver_stdout_colab_t4.txt) | Raw `mpirun -np 4` stdout from the Colab run |
| [`answers_q1_q6.md`](answers_q1_q6.md) | Source of the Q1–Q6 answers cell |
| [`make_notebook.py`](make_notebook.py) | Builds the notebook from the solver file (`--answers` refreshes only the Q1–Q6 cell and keeps outputs) |

## How each required mechanism is implemented

| Invariant | Implementation in `distributed_cuda_solver.py` |
|---|---|
| 1. Identity binding | `derive_student_seed(user, uuid)` = first 64 bits of SHA-256(`"user::uuid"`); the UUID comes from `nvidia-smi` on rank 0 and is broadcast to all ranks |
| 2. Distributed ownership | `initialize_subdomains` builds κ, u₀ and S on rank-local `(258, 258)` arrays from analytic global coordinates; no global array exists and no gather is ever called. The per-rank table is built with `Allreduce` of small metadata vectors |
| 3. Shared-memory tiling | `tiled_stencil_kernel`: 16 × 16 threads, `cuda.shared.array((18, 18), float64)`, centre + edge-thread apron loads, `cuda.syncthreads()`, then the flux-form update. The bounding box `(r_start, r_end, c_start, c_end)` selects the interior or a ring strip |
| 4. Strided halo packing | `pack_halos_kernel` / `unpack_halos_kernel` (1D grids) move rows and the strided columns into contiguous `(4, 256)` device buffers |
| 5. Dual-stream overlap | `step_overlapped`: Stage A on `stream_boundary` → host sync → `Irecv`×4 + `Isend`×4 → interior kernel on `stream_interior` → `Waitall` → async H2D → unpack → 4 ring kernels on `stream_boundary` |

Physics: κ(x,y) = 0.16 + 0.08·sin(2π(f_x x + φ_x))·cos(2π(f_y y + φ_y)) ∈ [0.08, 0.24]. The initial condition is three
Gaussian pulses, and S is a zero-net source/sink dipole. All of these parameters are drawn from `INSTANCE_SEED` with
`numpy.random.PCG64`. Outer walls are Dirichlet `u = 0` through `MPI.PROC_NULL` neighbours.

## Verification performed

1. **Colab T4 run (official):** every check in notebook cell 5 printed `OK`, and the solver printed `ALL TASKS COMPLETE: YES`.
2. **Integrity:** the manifest's `manifest_sha256` recomputes from its contents, and its `solver_sha256` equals the SHA-256 of
   the committed `distributed_cuda_solver.py`.
3. **Pre-flight correctness test (local, before Colab):** 4 MS-MPI ranks under `NUMBA_ENABLE_CUDASIM=1` on a 40 × 40 grid.
   Every rank's subdomain matched a single-array NumPy implementation of the same scheme to ≤ 1.1e-16, and serialized vs.
   overlapped parity was exactly 0. This check covers the halo exchange, partial tiles and ring/interior boxes.
4. **Physics cross-check:** the cubic-damping budget `−λ∫u³·100Δt` predicts ΔE = −1.956e-6 over steps 0→100. The run measured
   −1.922e-6, a 1.7 % difference (see Q6).

## Reproduction

On Colab, choose Runtime → Change runtime type → **T4 GPU**, upload `mpi_cuda_stencil_lab.ipynb`, then Runtime → **Run all**.
Cell 1 installs `mpi4py`, `numba` and `numba-cuda[cu12]`. The launcher cell uses the instructor's
`mpirun --allow-run-as-root --oversubscribe -np 4` command. When `mpirun` is absent, as on Windows with MS-MPI, it falls back
to `mpiexec -n 4`. The seed, and therefore every number above, changes with the GPU UUID of the session.

> Submission protocol note: the assignment asks for a separate repository named `mpi-cuda-stencil-Stierlitzx` with
> `mpi_cuda_stencil_lab.ipynb`, `distributed_cuda_solver.py` and `submission_manifest.json` at its root. This folder holds
> the same three files.
