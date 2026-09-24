# CSS314 Laboratory — The Amdahl Reality Gap

- **Student:** Ayazgaliyev Aibek
- **Student ID:** 230103341
- **Group:** 06-P, 02-N
- **Practicum Date & Time (Slot):** 24.09.2026, 12:20 local time (UTC+05:00)

## Workload

The Student ID workload formula gives:

```text
N = 10,000,000 + (341 × 1,000) = 10,341,000
```

For every integer in `[1, N]`, the benchmark computes the Collatz stopping time, the maximum stopping time, the number of values requiring more than 100 steps, and a checksum modulo `1,000,000,007`.

## Benchmark machine

- Lenovo 82Y9
- Windows 11 Pro 64-bit
- AMD Ryzen 5 7640HS
- 6 physical cores / 12 logical processors
- 64-byte L1 data-cache line
- MinGW GCC 6.3.0 with OpenMP

Complete hardware output is in [`src/hw_info.txt`](src/hw_info.txt).

## Project layout

```text
src/
├── collatz.c                 # C/OpenMP benchmark implementation
├── analyze_results.py        # CSV aggregation, Amdahl fitting, and plot generation
├── cache_line_probe.c        # CPUID cache-line verification utility
├── hw_info.txt               # Hardware and benchmark environment
├── results.csv               # Merged raw benchmark dataset
├── results_summary.csv       # Computed averages, speedups, gaps, and throughput
├── analysis_summary.txt      # Text summary of measured results
├── speedup_plot.png          # Empirical, Amdahl, and ideal speedup plot
├── raw/                      # Per-phase raw CSV files
└── logs/                     # Console logs from benchmark execution
```

`src/Main.kt` is an earlier unrelated Kotlin exercise and is not used by this laboratory.

## Build and run

From the repository root:

```powershell
gcc -O2 -fopenmp -std=c11 -Wall -Wextra -Wpedantic src/collatz.c -o src/collatz.exe
```

Run from `src` so generated CSV files remain with the source:

```powershell
cd src
.\collatz.exe all
python analyze_results.py
```

Run 1 of every configuration is a warmup. The analysis script averages Runs 2 and 3.

## Verified correctness results

All 42 benchmark runs produced the same logical output:

| Item | Result |
|---|---:|
| Checksum | 608,547,103 |
| Maximum stopping time | 685 steps |
| Values requiring more than 100 steps | 8,197,974 |

## Headline measured results

Sequential baseline:

```text
T_seq = 4.147500 seconds
```

Amdahl fitting from two threads:

```text
S_emp(2) = 1.901651
p = 0.948282
```

| Threads | Average time | Empirical speedup | Amdahl fit | Reality gap |
|---:|---:|---:|---:|---:|
| 1 | 4.059000 s | 1.0218× | 1.0000× | −0.0218 |
| 2 | 2.181000 s | 1.9017× | 1.9017× | 0.0000 |
| 4 | 1.144500 s | 3.6239× | 3.4627× | −0.1611 |
| 6 | 0.829500 s | 5.0000× | 4.7672× | −0.2328 |
| 8 | 0.633000 s | 6.5521× | 5.8736× | −0.6785 |
| 12 | 0.567500 s | 7.3084× | 7.6487× | +0.3403 |

The complete tables, including false-sharing and scheduling experiments, are in [`src/analysis_summary.txt`](src/analysis_summary.txt) and [`src/results_summary.csv`](src/results_summary.csv).

![Speedup plot](src/speedup_plot.png)
