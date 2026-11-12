# Performance Evaluation and Optimization in Parallel Computing

### Solo Python Systems Lab · 12 November 2026 · 11:40

**Ayazgaliyev Aibek · Student ID `230103341`**

Pure-Python multiprocessing experiments for shared-memory contention, empirical
scaling, roofline measurements, and skewed-workload scheduling. The scripts use
only the Python standard library and are written for Windows `spawn`.

## Results at a glance

| Experiment | Measured result | Main finding |
|---|---:|---|
| Hardware calibration | 88.0587 ns timer-call latency | 6 physical / 12 logical processors |
| False-sharing stride sweep | 754.682 ms at stride 1, `p=6` | Process launch and OS noise dominate this run; no reliable cache-line cliff was isolated |
| Strong scaling | **5.5823x** at 12 workers | Fixed workload falls from 2995.906 ms to 536.681 ms |
| Weak scaling | 564.216–939.685 ms | Runtime grows as the per-worker workload is increased |
| Roofline copy bandwidth | **2.1975 GB/s** | Standard-library byte-copy ceiling on this run |
| Roofline compute ceiling | **0.1677 GFLOP/s** | Scalar Python arithmetic is interpreter-bound |
| Dynamic scheduling | **14343.952 ms**, chunk 64 | Best dynamic wall time; imbalance factor lambda = 1.022 |

The complete raw capture is [`full_run_2026-11-12.txt`](full_run_2026-11-12.txt).
The measured analysis is [`lab_report_230103341.md`](lab_report_230103341.md).

## Hardware and runtime baseline

| Property | Recorded value |
|---|---|
| CPU model | AMD Ryzen 5 7640HS |
| Physical cores / logical threads | **6 / 12** |
| Operating system | Windows 11 |
| Python runtime | CPython 3.14.3 |
| Multiprocessing start method | `spawn` |
| Timer | `time.perf_counter_ns()` |
| Dependencies | Python standard library only |

The current Phase 0 script records processor counts and timer calibration. It
does not yet query L1d, L2, L3, or cache-line size directly through Windows;
those values are intentionally marked unavailable rather than fabricated.

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
| 1 | 8 bytes | 754.682 | 359.697 | 31.801 |
| 2 | 16 bytes | 1383.462 | 408.856 | 17.348 |
| 4 | 32 bytes | 1254.488 | 30.434 | 19.131 |
| 8 | 64 bytes | 1386.869 | 412.643 | 17.305 |
| 16 | 128 bytes | 1271.015 | 93.098 | 18.883 |
| 32 | 256 bytes | 1414.416 | 186.283 | 16.968 |

The run does not support claiming a clean cache-line boundary: stride 1 was
faster than the larger strides at `p=6`, and process creation, scheduling, and
Windows variance are substantial relative to the Python loop. At `p=12`,
stride 1 measured 1094.568 ms and stride 16 measured 1087.476 ms.

## M2 — Strong and weak scaling

[`task2_scaling.py`](task2_scaling.py) runs a serial trigonometric stage and a
pool-based floating-point kernel.

| Workers | Strong ms | Weak ms |
|---:|---:|---:|
| 1 | 2995.906 | 564.216 |
| 2 | 1540.211 | 467.837 |
| 3 | 1128.215 | 549.988 |
| 4 | 931.393 | 555.418 |
| 5 | 739.682 | 563.870 |
| 6 | 762.785 | 585.265 |
| 7 | 705.041 | 730.721 |
| 8 | 843.016 | 736.664 |
| 9 | 634.297 | 755.296 |
| 10 | 475.786 | 840.668 |
| 11 | 481.871 | 979.176 |
| 12 | 536.681 | 939.685 |

Observed strong speedup at 12 workers is **5.5823x**. The measured serial
fraction is **0.070063**. Weak-scaling time increases after the physical-core
boundary as the workload grows and logical processors contend for resources.

## M3 — Empirical roofline probe

[`task3_roofline.py`](task3_roofline.py) measures shared-buffer copying and an
unrolled scalar floating-point loop across six workers.

| Quantity | Value |
|---|---:|
| Sustained copy bandwidth | **2.1975 GB/s** |
| Sustained compute throughput | **0.1677 GFLOP/s** |
| Measured ridge point | **0.076331 FLOP/byte** |

These values characterize the supplied Python microbenchmarks, not the
processor's theoretical vector peak. The current implementation does not yet
contain the assignment's three GEMM variants, ten-point tile-size sweep,
numerical equivalence check, or GEMM roofline placement.

## M4 — Skewed scheduling and load imbalance

[`task4_scheduling.py`](task4_scheduling.py) evaluates contiguous static,
cyclic static, and shared-counter dynamic scheduling over the triangular
workload `K(i) = 1 + floor(i / 60)^2`.

| Strategy | Chunk | Wall ms | Imbalance lambda |
|---|---:|---:|---:|
| Contiguous block static | — | 24854.597 | not retained |
| Cyclic static | — | 15319.219 | not retained |
| Dynamic | 1 | 16037.395 | 1.002 |
| Dynamic | 4 | 15807.804 | 1.004 |
| Dynamic | 16 | 14585.968 | 1.011 |
| Dynamic | 64 | **14343.952** | **1.022** |
| Dynamic | 256 | 15049.897 | 1.118 |
| Dynamic | 1024 | 18520.081 | 1.433 |

Chunk 64 is the best measured dynamic configuration. Small chunks reduce
tail skew but increase shared-counter synchronization; large chunks reduce
locking overhead while leaving a heavier final tail assigned to fewer workers.

## Verification hashes

These values are copied verbatim from the full run:

```text
VERIFY_HASH[PHASE0]: CF673008A43F97AF (P_log=12)
VERIFY_HASH[PHASE1]: 97ED22DCE41CE521 (P_log=12)
VERIFY_HASH[PHASE2]: 606C7130DF4D8A61 (P_log=12)
VERIFY_HASH[PHASE3]: A224B1EDA0C2FDE6 (P_log=12)
VERIFY_HASH[PHASE4]: 81849DF6007B8D4A (P_log=12)
```

## Submission layout

```text
lecture_task_12.11.2026_11.40/
├── README.md
├── lab_report_230103341.md
├── full_run_2026-11-12.txt
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
py lab_runner.py *> full_run_YYYY-MM-DD.txt
```

The quick mode validates process creation and output shape. The full mode
performs the assignment-scale workload and can take several minutes. Results
are hardware-dependent; do not copy the recorded values into a new run.
