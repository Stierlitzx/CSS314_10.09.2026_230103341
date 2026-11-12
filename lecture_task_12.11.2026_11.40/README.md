# Performance Evaluation and Optimization in Parallel Computing

**Student:** 230103341  
**Session:** 12.11.2026, 11:40  
**Language:** Python 3 standard library only  
**Platform used for the recorded run:** Windows 11, CPython 3.14.3

This directory contains the five-phase parallel-computing systems lab:
hardware calibration, shared-memory contention, strong/weak scaling, a
hardware roofline probe, and skewed-workload scheduling.

## Run the lab

Use the short workload for a smoke test:

```powershell
py lab_runner.py --quick
```

Run the full hardware-bound experiments and capture their output:

```powershell
py lab_runner.py *> full_run_YYYY-MM-DD.txt
```

Individual phases can also be run separately:

```powershell
py task0_cal.py
py task1_falsesharing.py
py task2_scaling.py
py task3_roofline.py
py task4_scheduling.py
```

All multiprocessing worker functions are defined at module scope so the
programs work with Windows `spawn`. Shared state uses only standard-library
`multiprocessing`, `shared_memory`, `ctypes`, and timing primitives.

## Phase coverage

| Phase | Program | Experiment |
|---|---|---|
| 0 | `task0_cal.py` | Logical/physical CPU count, timer calibration, Phase 0 verification hash |
| 1 | `task1_falsesharing.py` | Shared-memory stride sweep and process scaling at strides 1 and 16 |
| 2 | `task2_scaling.py` | Strong and weak scaling across worker counts with serial-stage measurement |
| 3 | `task3_roofline.py` | Shared-buffer copy bandwidth, scalar floating-point compute ceiling, ridge point |
| 4 | `task4_scheduling.py` | Block, cyclic, and dynamic chunk scheduling with worker-time imbalance |

## Recorded full run

Raw terminal output is stored in
[`full_run_2026-11-12.txt`](full_run_2026-11-12.txt), and the report is in
[`lab_report_230103341.md`](lab_report_230103341.md).

Measured headline values from that run:

| Measurement | Result |
|---|---:|
| Physical / logical CPUs | 6 / 12 |
| Phase 0 timer call latency | 88.0587 ns |
| Phase 2 strong speedup at 12 workers | 5.5823x |
| Phase 2 measured serial fraction | 0.070063 |
| Phase 3 copy bandwidth | 2.1975 GB/s |
| Phase 3 compute throughput | 0.1677 GFLOP/s |
| Phase 3 ridge point | 0.076331 FLOP/byte |
| Phase 4 block-static wall time | 24854.597 ms |
| Phase 4 cyclic-static wall time | 15319.219 ms |
| Best dynamic configuration | chunk 64, 14343.952 ms |
| Best dynamic load-imbalance factor | lambda = 1.022 |

## Verification hashes

These hashes correspond to the recorded full run and the machine's 12 logical
processors:

```text
VERIFY_HASH[PHASE0]: CF673008A43F97AF (P_log=12)
VERIFY_HASH[PHASE1]: 97ED22DCE41CE521 (P_log=12)
VERIFY_HASH[PHASE2]: 606C7130DF4D8A61 (P_log=12)
VERIFY_HASH[PHASE3]: A224B1EDA0C2FDE6 (P_log=12)
VERIFY_HASH[PHASE4]: 81849DF6007B8D4A (P_log=12)
```

## Important scope note

The captured run is an implementation and telemetry baseline. The current
Phase 3 script measures the roofline microbenchmarks but does not yet include
the assignment's three GEMM implementations, ten-point tile sweep, or
numerical equivalence table. Phase 0 also does not query cache sizes directly
from Windows. These unavailable values are marked as such in the report
rather than fabricated.

## Files

- `harness.py` — benchmark statistics and FNV-1a verification tags.
- `lab_runner.py` — sequential runner for all five phases.
- `task0_cal.py` — hardware and timer calibration.
- `task1_falsesharing.py` — shared-memory false-sharing experiment.
- `task2_scaling.py` — strong/weak scaling experiment.
- `task3_roofline.py` — bandwidth and compute-ceiling experiment.
- `task4_scheduling.py` — static and dynamic scheduling experiment.
- `lab_report_230103341.md` — report containing the measured analysis.
- `full_run_2026-11-12.txt` — raw terminal capture from the full run.
