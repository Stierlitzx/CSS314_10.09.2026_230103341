# Empirical Scalability Diagnostics — Results

**Student:** 230103341 · **Session:** 29.10.2026, 14:30 · **Branch:** `practice-230103341`

**Hardware:** AMD Ryzen 5 7640HS w/ Radeon 760M Graphics; 6 physical cores / 12 logical threads; Windows-11-10.0.22631-SP0; CPython 3.14.3; multiprocessing start method `spawn`.

## Required end-to-end measurements

| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | S4 | E4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A: Serial tail** | 9.580845 | 5.541189 | 3.339213 | 0.156722 | 0.131374 | 2.869× | 71.73% |
| **B: Lock contention** | 8.522733 | 26.682036 | 25.734139 | 5.261380 | 3.692627 | 0.331× | 8.28% |
| **C: Privatized state** | 0.931186 | 0.663127 | 0.442602 | 0.424262 | 0.300413 | 2.104× | 52.60% |

- **A — serial path:** e(4)/e(2) = **0.838**, within the required [0.80, 1.25] interval. The absolute values include process startup/coordination, and are not guaranteed to match the handout's illustrative 0.10–0.14 range.
- **B — contention:** T4/T1 = **3.02×**, so adding workers makes this implementation slower. Both e values are very large, but **e(4) < e(2)** in this run: the rubric's increasing-trend condition was not observed.
- **C — privatization:** T4 < T1 and speedup is **2.10×**; end-to-end four-worker efficiency is **52.60%**, below the requested 70%. The original pool-startup-inclusive measurements remain the primary results.

## Supplementary C: prestarted-pool diagnostic

| P | Runtime (s) | Speedup | Efficiency | e(P) |
|---:|---:|---:|---:|---:|
| 1 | 0.828145 | 1.000× | 100.00% | — |
| 2 | 0.506307 | 1.636× | 81.78% | 0.222749 |
| 4 | 0.271106 | 3.055× | 76.37% | 0.103154 |

Every worker is confirmed ready before this timer starts. The timer still includes task dispatch, private loops, IPC and the final reduction. Pool startup and shutdown are excluded; `lifecycle_ns` retains their full cost separately. The improvement supports startup/coordination overhead as a contributor; it does not isolate all hardware or scheduling effects. Even this diagnostic does not establish e≤0.03.

## Raw timing samples

Each cell is seconds. All trials are retained, with no outlier removal.

| Mode | P | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|---:|
| A | 1 | 9.828465 | 9.017757 | 9.580845 | 9.580845 |
| A | 2 | 5.541189 | 5.781032 | 5.319470 | 5.541189 |
| A | 4 | 3.409957 | 3.232968 | 3.339213 | 3.339213 |
| B | 1 | 8.820818 | 8.522733 | 8.365365 | 8.522733 |
| B | 2 | 26.682036 | 28.389830 | 22.725054 | 26.682036 |
| B | 4 | 27.665128 | 25.734139 | 23.994564 | 25.734139 |
| C | 1 | 1.109726 | 0.931186 | 0.926098 | 0.931186 |
| C | 2 | 0.794536 | 0.578488 | 0.663127 | 0.663127 |
| C | 4 | 0.442602 | 0.460909 | 0.438834 | 0.442602 |
| C-steady | 1 | 0.828145 | 0.825575 | 1.270501 | 0.828145 |
| C-steady | 2 | 0.506307 | 0.520201 | 0.491069 | 0.506307 |
| C-steady | 4 | 0.204197 | 0.271106 | 0.324968 | 0.271106 |

## Method and interpretation

- A uses exactly 16 chunks, each summing `sin(i)*cos(i)` for i=1..1,999,999, preceded by `sleep(0.40)`. The specified pause is only **4.17%** of this host's T1; the comment claiming ~12% is machine-dependent.
- B performs exactly 2,000,000 increments under one explicit shared lock. The synchronized `Value('i', 0)` retains its default internal lock, as in the handout. The timings cover this combined synchronization cost, not isolated cache-coherence latency.
- C performs exactly 20,000,000 private increments and reduces the results once. B and C therefore have **different total work**; compare scaling within each experiment rather than treating their raw runtime ratio as a same-work speedup.
- Worker counts are 1, 2 and 4. Three trials rotate worker-count order to limit systematic order bias. Affinity and power settings are OS-managed. The baseline includes startup/shutdown, just like the supplied examples.
- Partitions preserve remainders for optional non-divisible totals. All full-size counts and all A chunk results were checked outside timing.
- Karp–Flatt is computed from **fixed total work** and each experiment's own T1. Values above one are retained; a diagnostic trend alone does not mathematically prove a unique hardware cause.
- The handout's illustrative B timing table is numerically inconsistent with its formula: T1=0.65, T2=1.10 and T4=2.40 imply e(2)=2.385 and e(4)=4.590. This report applies the stated formula to the measured values.

## Execution provenance

- **A:** 2026-10-29T14:40:56+0500 → 2026-10-29T14:41:56+0500.
- **B:** 2026-10-29T14:48:20+0500 → 2026-10-29T14:51:21+0500.
- **C:** 2026-10-29T14:53:16+0500 → 2026-10-29T14:53:23+0500.
- **C-steady:** 2026-10-29T14:54:42+0500 → 2026-10-29T14:54:50+0500.

The 29.10.2026 14:30 label is the scheduled session; captured timestamps are actual execution times. The folder was initially labelled 20.10 and renamed after the user's date correction; completed A measurements were preserved. An interrupted B attempt was not used in the recorded medians.

See [README.md](README.md) for plots and commands. Repository-root [RESULTS.md](../RESULTS.md) contains the required submission summary.
