# LAB REPORT — OpenMP Multi-Core Scaling in Python (Numba)

**Student:** 230103341 | **Date:** 01.10.2026
**CPU:** AMD Ryzen 5 7640HS (6 cores / 12 threads, Zen 4, SMT enabled)
**Software:** Python 3.14.3, NumPy 2.5.3, Numba 0.67.0 (multithreaded parallel engine)

---

## Table 1 — Student Result Scorecard

| Benchmark | Active Threads | Size / Samples | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|---|
| Ch1: Monte Carlo | 1 (base) | 120,000,000 | 2.2401 | 1.00x | 100.0% |
| Ch1: Monte Carlo | 2 | 120,000,000 | 1.3111 | 1.71x | 85.4% |
| Ch1: Monte Carlo | 4 | 120,000,000 | 0.9896 | 2.26x | 56.6% |
| Ch1: Monte Carlo | 8 | 120,000,000 | 0.5781 | 3.87x | 48.4% |
| Ch1: Monte Carlo | Max (12) | 120,000,000 | 0.5822 | 3.85x | 32.1% |
| Ch2: Mandelbrot (Rows) | Max (12) | 2500x2500, 1000 it | 1.441 | N/A | N/A |
| Ch2: Mandelbrot (Cols) | Max (12) | 2500x2500, 1000 it | 1.397 | N/A | N/A |
| Ch3: Heat Stencil (f64) | Max (12) | 1500x1500 x 300 steps | 0.438 | N/A | N/A |
| Ch3: Heat Stencil (f32) | Max (12) | 1500x1500 x 300 steps | 0.333 | N/A | N/A |

Throughput: **1540.59 Mcells/s (f64)**, **2026.62 Mcells/s (f32)**. π estimate ≈ 3.141739.

---

## Challenge 1 — Core Speedometer & Amdahl's Law

**B. Why the warm-up call is strictly required.**
The first call to an `@njit(parallel=True)` function triggers LLVM compilation of the whole
function (type inference, lowering, parallel-scheduling code generation). This compile step takes
seconds — orders of magnitude longer than the actual execution. Without the warm-up, T1 would be
dominated by compilation time, making the measured "1-thread baseline" meaningless and every
speedup/efficiency number invalid. The warm-up forces Numba to compile and cache the machine code
once, so all subsequently timed runs execute pure machine code.

**C. Efficiency scaling & architectural factors.**
Efficiency did NOT stay at 100%: it fell from 100% (1T) → 85.4% (2T) → 56.6% (4T) → 32.1% (12T),
with a best speedup of only ~3.87x on 12 threads. Causes:

1. **SMT/hyperthreading:** this CPU has 6 physical cores but 12 logical threads. Threads 7–12
   share execution units with threads 1–6; Monte Carlo is ALU/FLOP-bound, so SMT siblings add
   almost no throughput. 8T→12T actually got *slightly slower* (0.578 → 0.582 s) due to added
   scheduling contention and per-thread RNG state pressure.
2. **Memory-bus contention:** all 12 threads hammer shared L3/DRAM and random-number generation
   touches per-thread state caches.
3. **Amdahl's Law:** the small serial fraction (loop setup, final reduction merge, dispatch)
   cannot be parallelized, capping maximum speedup below N.
4. **Turbo-frequency effects:** all-core boost clock is lower than single-core boost, so
   per-core speed drops as thread count rises.

**D. Why `inside_circle += 1` under `prange` is safe.**
Numba recognizes the `+=` as a *reduction variable*: each thread accumulates into a private,
thread-local copy inside its chunk of iterations, and after the parallel region Numba merges all
private copies into the final result with `+`. This is exactly the semantics of OpenMP's
`#pragma omp parallel for reduction(+:inside_circle)` — the implicit construct Numba generates.
No lock is needed, so there is no race and no lost updates.


---

## Challenge 2 — Load Imbalance & Dynamic Scheduling

**A. Rows vs Columns.**
Results: rows = 1.441 s, cols = 1.397 s — nearly identical on this machine (cols ~3% faster).
Both variants perform identical total work, and Numba's parallel scheduler already uses dynamic
work-stealing chunking, so the per-iteration imbalance is absorbed either way; the small column
advantage comes from slightly better reuse of the destination image in cache.

**B. C-contiguous memory layout & cache lines.**
NumPy arrays are row-major: `img[r, c]` and `img[r, c+1]` are adjacent in memory. In the
**row-parallel** version, each thread's inner loop writes a contiguous 2500 × 4-byte = 10 KB
row — perfectly sequential, filling whole 64-byte cache lines, ideal spatial locality. In the
**column-parallel** version, a thread writes `img[r, c]` while jumping a full row stride (10 KB)
between successive writes: each 4-byte write touches a *different* cache line, wasting ~93% of
every 64-byte line transferred, and two threads writing into the same cache line at row
boundaries causes false sharing / coherence traffic.

**C. Load imbalance.**
Mandelbrot work per pixel is wildly non-uniform: points near the top of the view (low `r`,
negative imaginary part) escape in a few iterations, while the deep interior around the main
cardioid (center rows) runs all `max_iter = 1000` iterations. If thread 0 gets the top 20% of
rows it finishes almost instantly and idles, while the thread holding the center 20% of rows
does ~10–100x more work and becomes the critical path. With **dynamic scheduling**, iterations
(rows) are handed out in small chunks from a shared queue on demand: an idle thread immediately
pulls the next chunk, so work is redistributed at runtime and the queue drains balanced even
though per-row cost is unpredictable. The equivalent explicit OpenMP clause is
`schedule(dynamic, chunk)`.

**D.** `mandelbrot_output.png` is attached in the repository (generated by the benchmark script).

---

## Challenge 3 — Stencil Computation & Memory Bandwidth

**A. Benchmark results:** 0.438 s for 300 steps on 1500×1500 f64 grid = **1540.59 Megacells/sec**.

**B. float64 → float32.**
Runtime dropped 0.438 s → 0.333 s = **1.32x faster** (throughput 1540 → 2026 Mcells/s).
Why: each stencil step must read a cell plus its 4 neighbors and write 1 cell — roughly
5 loads + 1 store per cell per step, i.e. ~48 bytes moved per cell in f64 vs ~24 bytes in f32.
The arithmetic itself (one fused multiply-add per cell) is trivial compared to the memory
transfer; halving the footprint halves the bytes the DRAM bus must supply per unit of work,
raising the compute rate the memory system can feed. The measured speedup (1.32x) is below the
theoretical 2x because cache-line granularity (64-byte lines) and L2/L3 residency mean bus
transactions are not exactly halved.

**C. Memory Roofline Model.**
Every core has huge arithmetic throughput, but all cores share one memory hierarchy: the same
L3 and the same DRAM bus. A 5-point stencil re-reads each cell ~5x with almost no register-level
reuse, so arithmetic intensity is ~0.17 FLOP/byte — deep in the memory-bound roofline region.
Once aggregate core demand (N cores × per-core bandwidth request) exceeds the DRAM bandwidth
ceiling (~50–100 GB/s on a laptop), extra cores just queue on the bus: runtime is pinned at
`bytes_moved / DRAM_bandwidth` regardless of core count, so doubling cores yields near-zero
extra speedup instead of 2x. Speedup saturates at
`min(N × per-core compute, DRAM_bandwidth / bytes_per_cell)` — the memory wall.

---

*Artifacts: `lab_openmp_benchmark.py` (benchmark code), `benchmark_results.txt` (raw output),
`mandelbrot_output.png` (fractal render).*

