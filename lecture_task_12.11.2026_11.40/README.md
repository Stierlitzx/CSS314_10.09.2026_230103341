# Performance Evaluation and Optimization in Parallel Computing

This folder contains the zero-dependency Python implementation of the five-phase
systems lab for student `230103341`.

Run a short smoke test with `py lab_runner.py --quick`. The full experiments
are intentionally hardware-bound and can take substantially longer; run each
phase directly when collecting the final measurements:

```powershell
py task0_cal.py
py task1_falsesharing.py
py task2_scaling.py
py task3_roofline.py
py task4_scheduling.py
```

All worker targets are module-level functions, shared state uses standard
library multiprocessing primitives, and every phase emits its own
`VERIFY_HASH` line. The measurements in `lab_report_230103341.md` must be
captured from the same machine and run rather than entered manually.
