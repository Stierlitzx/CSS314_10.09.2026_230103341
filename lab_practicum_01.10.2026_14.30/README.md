# Lab Practicum — OpenMP Multi-Core Scaling in Python (01.10.2026, started 14:30)

Numba `@njit(parallel=True)` / `prange` benchmarks of the GIL-free multithreaded
parallel engine across three challenges, plus a load-imbalance and memory-bandwidth study.

**Student:** Ayazgaliyev Aibek, 230103341
**CPU:** AMD Ryzen 5 7640HS — 6 cores / 12 threads (SMT), Zen 4
**Software:** Python 3.14.3, NumPy 2.5.3, Numba 0.67.0, Matplotlib 3.11.2 (Windows 11)

## Structure

```
lab_practicum_01.10.2026_14.30/
├── src/
│   └── lab_openmp_benchmark.py   # all three challenges in one script (+ float32 variant)
└── result/
    ├── benchmark_results.txt     # raw console output of the run
    ├── mandelbrot_output.png     # exported fractal render (2500x2500)
    └── LAB_REPORT.md             # filled Table 1 scorecard + full answers
```

## How to run

```bash
pip install numpy numba matplotlib
python src/lab_openmp_benchmark.py
```

Outputs (render image + raw results) are written automatically into `result/`.
No C/C++ compiler is required — Numba JIT-compiles everything on first call
(the script includes JIT warm-up calls so compile time is excluded from timings).

## Results summary

| Benchmark | Threads | Size / Samples | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|---|
| Ch1: Monte Carlo π | 1 (base) | 120,000,000 | 2.2401 | 1.00x | 100.0% |
| Ch1: Monte Carlo π | 2 | 120,000,000 | 1.3111 | 1.71x | 85.4% |
| Ch1: Monte Carlo π | 4 | 120,000,000 | 0.9896 | 2.26x | 56.6% |
| Ch1: Monte Carlo π | 8 | 120,000,000 | 0.5781 | **3.87x (best)** | 48.4% |
| Ch1: Monte Carlo π | 12 (max) | 120,000,000 | 0.5822 | 3.85x | 32.1% |
| Ch2: Mandelbrot (rows) | 12 | 2500x2500, 1000 it | 1.441 | — | — |
| Ch2: Mandelbrot (cols) | 12 | 2500x2500, 1000 it | 1.397 | — | — |
| Ch3: Heat stencil f64 | 12 | 1500x1500 x 300 steps | 0.438 | — | — |
| Ch3: Heat stencil f32 | 12 | 1500x1500 x 300 steps | 0.333 | — | — |

- π estimate: **3.141739**
- Ch3 throughput: **1540.59 Mcells/s (f64)** → **2026.62 Mcells/s (f32)** = **1.32x** faster

## Key takeaways

1. **Amdahl's Law + SMT:** best speedup 3.87x on 12 threads — the 6 physical cores do
   the real work; SMT siblings, memory-bus contention and the lower all-core boost clock
   cap efficiency at ~32%.
2. **Reductions:** `inside_circle += 1` under `prange` maps to OpenMP
   `reduction(+:...)` — per-thread private counters merged after the parallel region.
3. **Load imbalance:** Mandelbrot row/column decomposition performs nearly identically
   because Numba's scheduler already uses dynamic work-stealing; column writes waste
   cache-line bandwidth due to row-major layout.
4. **Memory wall:** the 5-point stencil is bandwidth-bound (~0.17 FLOP/byte); halving the
   element size gave only 1.32x, and extra cores beyond DRAM saturation add no speedup.
