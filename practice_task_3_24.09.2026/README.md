# CSS314 — Practice Task 3 (24.09.2026): Shared-Memory Concurrency & OpenMP Paradigms Lab

Implementations of Labs 1–5 from the *Laboratory Practice Manual: Shared-Memory
Concurrency & OpenMP Paradigms* (Instructor – Sufyan bin Uzayr).

**Language edition used:** Java 21 (OpenJDK). The Java edition was chosen because
genuine OS threads (no GIL) make race conditions, false sharing, lock contention
and work-stealing directly observable. All programs map one-to-one onto the
OpenMP directives listed in the manual (`ForkJoinPool` ≈ `#pragma omp parallel`,
parallel streams ≈ `parallel for reduction`, `synchronized` ≈ `critical`,
`AtomicInteger` work dispenser ≈ `schedule(dynamic,chunk)`,
`RecursiveAction.invokeAll` ≈ `task` + `taskwait`).

Earlier lecture work (Collatz OpenMP benchmark in C) is preserved in
[`../lecture_task_10.09.2026/`](../lecture_task_10.09.2026/).

## Hardware Specification (Section I)

| Property | Value |
|---|---|
| CPU | AMD Ryzen 5 7640HS w/ Radeon 760M Graphics |
| Physical cores | 6 |
| Logical threads | 12 (SMT-2) |
| L2 cache | 6 MB (1 MB/core) |
| L3 cache | 16 MB |
| OS | Windows 11 (win32) |
| Runtime | OpenJDK 21.0.2, Python 3.14.3 (plotting only) |

## Repository Layout

```
lab1/ForkJoinLab1.java           Fork-join model, thread scoping, oversubscription
lab2/PiIntegrationLab2.java      Pi integration: race / critical / reduction
lab3/MandelbrotSchedulerLab3.java Static vs. dynamic loop scheduling (Mandelbrot)
lab4/FalseSharingLab4.java       Cache-line false sharing vs. padding vs. local acc.
lab5/ParallelMergeSortLab5.java  Recursive task-based parallel merge sort
labN/results/                    Raw CSV/TXT benchmark data per lab
plots/                           Generated figures (make_plots.py)
RESULTS.md                       Measured results, tables and discussion
```

## Build & Run (Section II — Methodology)

All timers use `System.nanoTime()`; each benchmark is preceded by JIT warm-up
runs. Trials per configuration are noted in `RESULTS.md`. Use the US locale flags
so CSV decimal separators are dots:

```powershell
# Compile all labs
javac lab1\ForkJoinLab1.java lab2\PiIntegrationLab2.java lab3\MandelbrotSchedulerLab3.java lab4\FalseSharingLab4.java lab5\ParallelMergeSortLab5.java

# Lab 1 (modes: demo | sweep | load)
java "-Duser.language=en" "-Duser.country=US" -cp lab1 ForkJoinLab1 demo
java "-Duser.language=en" "-Duser.country=US" -cp lab1 ForkJoinLab1 sweep
java "-Duser.language=en" "-Duser.country=US" -cp lab1 ForkJoinLab1 load

# Lab 2 (modes: race | critical | scale | all)
java "-Duser.language=en" "-Duser.country=US" -cp lab2 PiIntegrationLab2 all

# Lab 3 (modes: matrix | imbalance)
java "-Duser.language=en" "-Duser.country=US" -cp lab3 MandelbrotSchedulerLab3 matrix
java "-Duser.language=en" "-Duser.country=US" -cp lab3 MandelbrotSchedulerLab3 imbalance

# Lab 4
java "-Duser.language=en" "-Duser.country=US" -cp lab4 FalseSharingLab4

# Lab 5
java "-Duser.language=en" "-Duser.country=US" -cp lab5 ParallelMergeSortLab5

# Regenerate plots (requires numpy + matplotlib)
python make_plots.py
```
