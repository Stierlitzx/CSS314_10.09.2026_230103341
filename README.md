# CSS314 — Parallel Programming (Ayazgaliyev Aibek, 230103341)

Repository for CSS314 Parallel Programming course tasks.

## Contents

| Folder | Description |
|---|---|
| [`lecture_cuda_Lab01_08.10.2026_11.30/`](lecture_cuda_Lab01_08.10.2026_11.30/READM.md) | **CUDA Lecture Lab 01 (08.10.2026, started 11:30)** - CUDA Python experiments executed on the local RTX 4060 Laptop GPU. Includes an illustrated report, executed notebook, filled task sheet, raw measurements, and verification checksum. |
| [`lecture_task_24.09.2026_11.30/`](lecture_task_24.09.2026_11.30/) | **Lecture task (24.09.2026, started 11:30)** — Collatz sequence OpenMP benchmark in C: false sharing, scheduling and strong-scaling experiments. |
| [`practice_task_3_24.09.2026_14.30/`](practice_task_3_24.09.2026_14.30/) | **Practice task 3 (24.09.2026, started 14:30)** — Shared-Memory Concurrency & OpenMP Paradigms labs 1–5 (Java 21): fork-join thread teams, Pi reductions, Mandelbrot loop scheduling, false sharing, task-based parallel merge sort. Includes raw benchmark CSVs, plots and a full results write-up (`RESULTS.md`). |
| [`practice_cuda_Lab02_08.10.2026_14.30/`](practice_cuda_Lab02_08.10.2026_14.30/README.md) | **CUDA Practice Lab 02 (08.10.2026, started 14:30)** — Advanced CUDA Python: warp-divergence microbenchmark, 1D boundary stencil with halo clamping, grid-stride vector scaling (16,384 threads over 100,000 elements), and 2D Sobel-X convolution. Illustrated README, autonomous verification suite, and integrity token `F8D4B55BD129CA58D4CF`. |
| [`lab_practicum_01.10.2026_14.30/`](lab_practicum_01.10.2026_14.30/) | **Lab practicum (01.10.2026, started 14:30)** — OpenMP multi-core scaling in Python via Numba `prange`: Monte Carlo π speedup sweep, Mandelbrot axis decomposition & load imbalance, memory-bandwidth-bound heat stencil. Best measured speedup **3.87×**; float32 stencil **1.32×** over float64 (1540 → 2027 Mcells/s). Source in `src/`, artifacts and report in `result/`. |

Hardware used for the completed CPU benchmarks: AMD Ryzen 5 7640HS, 6 cores / 12 threads,
L2 6 MB, L3 16 MB, Windows 11.
