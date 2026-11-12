# Performance Evaluation and Optimization in Parallel Computing

### Solo Python Systems Lab · 12 November 2026 · 11:40

**Ayazgaliyev Aibek · Student ID `230103341`**

Pure-Python multiprocessing experiments for shared-memory contention, empirical
scaling, roofline measurements, and skewed-workload scheduling. The scripts use
only the Python standard library and are written for Windows `spawn`.

## Results at a glance

| Experiment | Measured result | Main finding |
|---|---:|---|
| Hardware calibration | 167.7681 ns timer-call latency | 6 physical / 12 logical processors |
| False-sharing stride sweep | 1403.072 ms at stride 1, `p=6` | 64-byte separation was fastest in this run |
| Strong scaling | **7.3216x** at 12 workers | Fixed 1.2M-element workload |
| Weak scaling | 1344.886–1900.783 ms | Runtime grows beyond six physical cores |
| Roofline copy bandwidth | **2.3236 GB/s** | 64 MB shared-buffer copy benchmark |
| Roofline compute ceiling | **0.1366 GFLOP/s** | 60M-FLOP scalar FMA benchmark |
| GEMM optimization | N=256, six tile sizes | Best measured tile: BS=64, 0.030422 GFLOP/s |
| Dynamic scheduling | **10257.402 ms**, chunk 1 | Best dynamic wall time; imbalance factor lambda = 1.004 |

The complete raw capture is [`full_run_2026-11-12.txt`](full_run_2026-11-12.txt).
The measured analysis is [`lab_report_230103341.md`](lab_report_230103341.md).

## Hardware and runtime baseline

| Property | Recorded value |
|---|---|
| CPU model | AMD Ryzen 5 7640HS w/ Radeon 760M Graphics |
| Physical cores / logical threads | **6 / 12** |
| L2 / L3 cache | **6144 KB / 16384 KB** |
| Cache-line size | **64 bytes** (x86-64 baseline) |
| Operating system | Windows 11 |
| Python runtime | CPython 3.14.3 |
| Multiprocessing start method | `spawn` |
| Timer | `time.perf_counter_ns()` |
| Dependencies | Python standard library only |

Phase 0 queries processor counts and L2/L3 values through Windows CIM and
records the x86-64 cache-line baseline. Windows CIM does not expose L1d size in
this script.

## M0 — Harness and verification

[`harness.py`](harness.py) provides:

- median, IQR, minimum, and maximum benchmark statistics;
- FNV-1a 64-bit verification tags;
- conservative physical-core detection for this Windows host.

[`lab_runner.py`](lab_runner.py) executes the five phases sequentially. Every
phase keeps worker functions at module scope and uses an `if __name__ ==
"__main__"` guard so Windows process spawning is safe.

## M1 — False sharing and shared-memory contention

[`task1_falsesharing.py`](task1_falsesharing.py) allocates a 4096-byte
`multiprocessing.shared_memory` segment and updates separate `ctypes.c_uint64`
slots from multiple processes. It measures strides of 1, 2, 4, 8, 16, and 32
words, followed by scaling at strides 1 and 16.

### Stride sensitivity at six workers

| Stride | Separation | Median ms | IQR ms | Mops/s |
|---:|---:|---:|---:|---:|
| 1 | 8 bytes | 1403.072 | 103.365 | 11.404 |
| 2 | 16 bytes | 1385.897 | 30.681 | 11.545 |
| 4 | 32 bytes | 1348.652 | 53.895 | 11.864 |
| 8 | 64 bytes | 1334.728 | 57.164 | 11.987 |
| 16 | 128 bytes | 1349.644 | 181.554 | 11.855 |
| 32 | 256 bytes | 1349.901 | 58.564 | 11.853 |

The best measured separation was 32 bytes, not a clean cache-line boundary.
At `p=12`, stride 1 measured 875.277 ms and stride 16 measured 914.189 ms.
The Karp-Flatt values are printed for every scaling point in the raw capture.

## M2 — Strong and weak scaling

[`task2_scaling.py`](task2_scaling.py) runs a serial trigonometric stage and a
pool-based floating-point kernel.

| Workers | Strong ms | Weak ms |
|---:|---:|---:|
| 1 | 7987.341 | 1671.923 |
| 2 | 3555.843 | 1849.839 |
| 3 | 2292.564 | 1502.625 |
| 4 | 2001.573 | 1544.008 |
| 5 | 1635.643 | 1462.344 |
| 6 | 1380.468 | 1344.886 |
| 7 | 1189.918 | 1430.807 |
| 8 | 1180.617 | 1620.975 |
| 9 | 1088.278 | 1663.532 |
| 10 | 992.521 | 1707.785 |
| 11 | 965.664 | 1831.083 |
| 12 | 1090.925 | 1900.783 |

Observed strong speedup at 12 workers is **7.3216x**. The measured serial
fraction is **0.021040**. Weak-scaling time increases after the physical-core
boundary as the workload grows and logical processors contend for resources.

## M3 — Empirical roofline probe

[`task3_roofline.py`](task3_roofline.py) measures shared-buffer copying and an
unrolled scalar floating-point loop across six workers.

| Quantity | Value |
|---|---:|
| Sustained copy bandwidth | **2.3236 GB/s** |
| Sustained compute throughput | **0.1366 GFLOP/s** |
| Measured ridge point | **0.058769 FLOP/byte** |

These values characterize the supplied Python microbenchmarks, not the
processor's theoretical vector peak. The same phase also runs naive `i-j-k`,
reordered `i-k-j`, and cache-blocked GEMM variants at
`N=256` and `BS = {8,16,32,64,96,128}`, checking blocked output against the
reordered output with a maximum absolute error threshold of `1e-9`.

## M4 — Skewed scheduling and load imbalance

[`task4_scheduling.py`](task4_scheduling.py) evaluates contiguous static,
cyclic static, and shared-counter dynamic scheduling over the triangular
workload `K(i) = 1 + floor(i / 40)^2` over 8,000 items.

| Strategy | Chunk | Wall ms | Imbalance lambda |
|---|---:|---:|---:|
| Contiguous block static | — | 22998.501 | not retained |
| Cyclic static | — | 11082.593 | not retained |
| Dynamic | 1 | **10257.402** | **1.004** |
| Dynamic | 8 | 10361.800 | 1.006 |
| Dynamic | 64 | 10720.637 | 1.065 |
| Dynamic | 512 | 14085.482 | 1.328 |

Chunk 1 is the best measured dynamic configuration. Small chunks reduce
tail skew but increase shared-counter synchronization; large chunks reduce
locking overhead while leaving a heavier final tail assigned to fewer workers.

## Verification hashes

These values are copied verbatim from the Minimal Lab full run:

```text
VERIFY_HASH[PHASE0]: 44E99E04AEEB6783 (P_log=12)
VERIFY_HASH[PHASE1]: 9B5DEA72F6947F37 (P_log=12)
VERIFY_HASH[PHASE2]: D5A10D4FC85C47E3 (P_log=12)
VERIFY_HASH[PHASE3]: CAC1A5BEC1704C15 (P_log=12)
VERIFY_HASH[PHASE4]: 64BE1E5AD5965F27 (P_log=12)
```

## Submission layout

```text
lecture_task_12.11.2026_11.40/
├── README.md
├── lab_report_230103341.md
├── minimal_full_2026-11-12.txt
├── harness.py
├── lab_runner.py
├── task0_cal.py
├── task1_falsesharing.py
├── task2_scaling.py
├── task3_roofline.py
└── task4_scheduling.py
```

## Reproduction

Run these commands from this directory:

```powershell
py lab_runner.py --quick
py lab_runner.py *> minimal_full_YYYY-MM-DD.txt
```

The quick mode validates process creation and output shape. The full mode
performs the assignment-scale workload and can take several minutes. Results
are hardware-dependent; do not copy the recorded values into a new run.
