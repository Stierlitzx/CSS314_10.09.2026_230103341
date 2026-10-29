<!-- practice-karp-flatt:230103341 -->
# Karp–Flatt Practice Results — 29.10.2026, 14:30

**Student:** 230103341 · **Session:** 29.10.2026, 14:30 · **Branch:** `practice-230103341`

**Hardware:** AMD Ryzen 5 7640HS w/ Radeon 760M Graphics; 6 physical cores / 12 logical threads; Windows-11-10.0.22631-SP0; CPython 3.14.3; multiprocessing start method `spawn`.

| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | S4 | E4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A: Serial tail** | 9.580845 | 5.541189 | 3.339213 | 0.156722 | 0.131374 | 2.869× | 71.73% |
| **B: Lock contention** | 8.522733 | 26.682036 | 25.734139 | 5.261380 | 3.692627 | 0.331× | 8.28% |
| **C: Privatized state** | 0.931186 | 0.663127 | 0.442602 | 0.424262 | 0.300413 | 2.104× | 52.60% |

- **A — serial path:** e(4)/e(2) = **0.838**, within the required [0.80, 1.25] interval. The absolute values include process startup/coordination, and are not guaranteed to match the handout's illustrative 0.10–0.14 range.
- **B — contention:** T4/T1 = **3.02×**, so adding workers makes this implementation slower. Both e values are very large, but **e(4) < e(2)** in this run: the rubric's increasing-trend condition was not observed.
- **C — privatization:** T4 < T1 and speedup is **2.10×**; end-to-end four-worker efficiency is **52.60%**, below the requested 70%. The original pool-startup-inclusive measurements remain the primary results.

The separately measured prestarted C pool reaches **3.05×** and **76.37%** efficiency. Its timing excludes startup/shutdown and is supplementary.

[Full report, images, raw samples and reproduction commands](practice_task_29.10.2026_14.30/README.md).
