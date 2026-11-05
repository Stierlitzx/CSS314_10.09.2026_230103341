# Task 22 — Weak-scaling benchmark

**Ayazgaliyev Aibek · 230103341**

## Method

I implemented a CPU-bound midpoint integration of 4 / (1 + x²) over [0, 1],
whose integral is π. I used Python multiprocessing rather than threads, so
each worker runs in its own interpreter.

For P workers and k points per worker:

```text
Total points = P × k
Scaled speedup S = serial time for P × k points / parallel time for P × k points
Scaled efficiency = S / P
Weak runtime efficiency = one-worker parallel time / P-worker parallel time
```

I measured a matching direct serial reference for every trial. Both versions
use identical chunks and the same final summation order. I checked agreement
with each other and with π outside the timed sections.

My parallel timer includes process creation, Windows spawn, calculation, pipes,
reduction, joining, and cleanup. Exactly P distinct worker PIDs are required.
There is no artificial sleep or synthetic serial fraction.

## Experiment

I used **1,000,000 points per worker**, **P = 1–12**, and **five fresh trials**
per count. I shuffled configuration order, alternated serial/parallel timing
order, and retained all outliers. The CSV and adjacent metadata record parameters,
timestamps, hardware, and completion status.

I obtained better scaled throughput, but runtime did not remain constant.
The one-worker parallel path still includes spawn overhead, so its speedup can
be below one. My fitted α is an effective overhead parameter, not a measured
serial-code fraction. [Task 23](../../analysis/task23_experimental_results.md)
contains my results.

## Reproduction

My standard-library benchmark requires Python 3.10+. From PowerShell:

```powershell
$root = 'C:\Users\aiblb\PycharmProjects\CSS314_10.09.2026_230103341\gustafson-law-practice'
py "$root\benchmarks\weak_scale_bench\benchmark.py" --output "$root\benchmarks\repeat.csv"
```

Defaults cover every reported logical processor. Optional flags are
`--workers 1,2,4,6,12 --k 1000000 --runs 5 --seed 230103341`.
Existing files are protected unless `--overwrite` is supplied; my retained
submission remains unchanged. [Full commands](../../README.md#reproduction).