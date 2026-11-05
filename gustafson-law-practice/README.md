# Gustafson's Law & Weak Scaling — Comprehensive Practice

**Ayazgaliyev Aibek · Student ID 230103341**

**Course:** Parallel & Distributed Computing Architecture

**Instructor:** Sufyan bin Uzayr · **Assessment:** Ungraded individual practice

**Submission location:** dedicated `gustafson-law-practice/` folder in the personal
course repository. This README is the root documentation for this practice;
the existing repository root README indexes it without replacing prior coursework.

## Contents

1. [Executive synthesis](#executive-synthesis)
2. [Mathematical rules and assumptions](#mathematical-rules-and-assumptions)
3. [Measured results](#measured-results)
4. [Reproduce and validate](#reproduce-and-validate)
5. [Task-by-task artifact index](#task-by-task-artifact-index)
6. [Supporting artifacts and provenance](#supporting-artifacts-and-provenance)
7. [Engineering guidance](#engineering-guidance)

## Executive synthesis

Gustafson's law answers **how much larger a useful problem can be solved in a
fixed time when more processors are available**. Amdahl's law answers **how much
faster a fixed useful problem can be solved**. Neither invalidates the other.
Using the same number for fractions measured on different machines compares
different workloads; converting the fractions for the same workload makes
the speedup expressions agree.

The 25 tasks include derivations, exact numerical calculations, original-source
historical reconstruction, communication/memory/energy limits, algorithm-specific
work growth, and an actual process-based weak-scaling experiment. The simulator
is analytical; the 60 benchmark observations are **real measurements**, not
simulated samples. The measured implementation does **not** achieve ideal fixed
runtime; the retained overhead and variability are part of the findings.

## Mathematical rules and assumptions

Let P be the number of processors. Distinguish:

- `s_A`: useful serial fraction of the **one-core fixed workload**.
- `alpha_P`: serial fraction of the **parallel execution** of a specified scaled
  workload. A family with approximately constant alpha is an assumption, not a
  theorem about every program.

| Model | Formula | What stays fixed |
|---|---|---|
| Amdahl | `S_A(P)=1/[s_A+(1-s_A)/P]` | Total useful workload |
| Gustafson | `S_G(P)=P-alpha(P-1)` | Ideal parallel elapsed time; useful work grows |
| Gustafson efficiency | `E_G(P)=1-alpha+alpha/P` | Same fixed-time assumptions |
| Fraction conversion, same workload | `s_A=alpha/[P-alpha(P-1)]` | Reference frame changes, not the work |
| Communication-aware useful speedup | `S=[A+PB]/[A+B+C(P)]` | Serial useful time A and per-worker useful time B |

For constant alpha, `lim E_G=1-alpha`. For constant `s_A>0`, `lim E_A=0`;
the perfectly parallel `s_A=0` exception has efficiency one. Added communication
is not automatically useful serial work; counting it in the numerator inflates
speedup. Extra processors do not remove an increasing critical path.

Selected exact handout results:

- Task 06, alpha=0.04: P=16 gives **15.4×** and **96.25%** efficiency;
  P=16,384 gives **15,728.68×** and **96.000244%** efficiency.
- Task 07: maximum alpha is **692/8191 ≈ 0.0844829691124** for
  S(8192)≥7500; approximately **8.4483%**.
- Task 08: at P=128 and numerical fraction 0.08, Amdahl is
  **3200/279 ≈ 11.469534×**, Gustafson **117.84×**. These are different-workload
  predictions, not inconsistent answers for the same experiment.
- Task 09: treating the 2% I/O/broadcast and 6% synchronization as non-scaling
  elapsed phases gives alpha=0.08: **471.12×** at P=512 and **1884.24×** at P=2048.
- Task 10: no unique grid dimension exists without **bytes per cell**, reserved
  memory and partition assumptions. The aggregate-capacity bound is
  `floor[(32*10^9*P/b)^(1/3)]` for decimal GB and b bytes per cell.
- Task 16: compute-bounded GEMM fixed time needs `N∝P^(1/3)`, not `sqrt(P)`;
  memory grows as `P^(2/3)` for its matrices under this expansion.
- Task 17: ideal cubic local domains have constant local surface/volume under
  weak scaling. A shrinking **global** outside-surface/volume ratio does not
  prove shrinking **interprocess** halo cost.

## Measured results

Measurement session: **5 November 2026, 14:45:08–14:47:03, UTC+05:00**;
the metadata preserves full timestamps. Host: Windows 11, AMD Ryzen 5 7640HS,
6 physical cores / 12 logical processors, CPython 3.14.3.

Synthetic CPU workload: midpoint integration of `4/(1+x*x)` over `[0,1]`,
**1,000,000 points per process**, every integer P=1…12, five trials per count.
Each observation has a separately measured, same-work **direct serial reference**.
Parallel timing includes allocation, spawn, computation, IPC, reduction and cleanup.

| Metric | P=1 | P=6 | P=11 | P=12 |
|---|---:|---:|---:|---:|
| Mean parallel time (s) | 0.361641 | 0.658308 | 0.698435 | 0.768045 |
| Mean scaled speedup | 0.527832 | 2.040563 | **3.020116** | 2.919229 |
| Scaled efficiency S/P | 52.7832% | 34.0094% | 27.4556% | 24.3269% |

The fitted law has **alpha_observed=0.784991225892**, RMSE **0.246365** and
descriptive R² **0.905843**. This is an **effective overhead parameter**, not
measured serial source-code work. Direct serial and spawned single-worker
paths differ, so the measured P=1 speedup need not be one; this model mismatch
is retained and explicitly discussed. No thermal-throttling or NUMA cause is
asserted without telemetry. The largest observed mean is not a statistically
established universal optimum with five noisy trials.

![Measured and theoretical scaling](plots/scaling_comparison.png)

[Vector PDF](plots/scaling_comparison.pdf) · [full statistics and regression](analysis/task23_experimental_results.md)
· [hardware audit and limitations](analysis/task24_hardware_audit.md).

## Reproduce and validate

Python **3.10+**; all simulation calculations, benchmarking, fitting and tests use
the standard library. Only plots require Matplotlib, already used by this repository.
Run from any directory in PowerShell, using the absolute checkout path:

```powershell
$root = 'C:\Users\aiblb\PycharmProjects\CSS314_10.09.2026_230103341\gustafson-law-practice'

# Optional plotting dependency (installed in the measured environment).
py -m pip install -r "$root\requirements-plots.txt"

# Analytical simulator: arbitrary comma-separated core counts and fractions.
py "$root\benchmarks\simulator.py" --cores 1,16,64,256,1024,4096,16384,65536 --fractions 0.001,0.04,0.08,0.20

# Regenerate the default committed 68-row simulation and overlays.
py "$root\benchmarks\simulator.py"

# Reanalyze the retained real data, regenerate plots and numeric report.
py "$root\benchmarks\fit_gustafson.py"
py "$root\benchmarks\report_results.py"

# Repeat real measurements under a NEW filename; defaults cover all host threads.
py "$root\benchmarks\weak_scale_bench\benchmark.py" --output "$root\benchmarks\repeat.csv"

# Correctness, actual-process smoke test, raw-data validation and artifact checks.
py "$root\benchmarks\test_practice.py"
py "$root\benchmarks\verify_artifacts.py"
```

On other systems, set the absolute root to that checkout and use the installed
Python command (often `python3`). The measured Windows `python` alias does not
resolve an interpreter; **`py` works**. `--no-plots` on the simulator/fitter
supports a fully standard-library workflow. The simulator enforces
`1≤P≤65,536` and `0.001≤fraction≤0.20`; the empirical fitted alpha is not restricted
to that *simulator input* interval. The benchmark refuses accidental overwrite;
`--overwrite` explicitly opts in. No dependency installation is needed to inspect
the committed tables or PDFs.

## Task-by-task artifact index

| Task | Deliverable | Topic |
|---|---|---|
| 01 | [Scaling models](theory/task01_scaling_models.md) | Strong vs weak; fixed size vs fixed time |
| 02 | [Derivation](theory/task02_derivation.md) | Equivalent one-core work, line by line |
| 03 | [Serial invariance](theory/task03_serial_invariance.md) | Reference-frame conversion |
| 04 | [Asymptotic limits](theory/task04_asymptotic_limits.md) | Efficiency proofs and boundary cases |
| 05 | [Historical nCUBE](theory/task05_historical_ncube.md) | Primary 1988 findings, distinct reported datasets |
| 06 | [Parametric table](calculations/task06_parametric_table.md) | Required six P values, alpha=0.04 |
| 07 | [Tolerable serial fraction](calculations/task07_tolerable_serial.md) | Exact 8192-core tolerance |
| 08 | [Cross-model divergence](calculations/task08_cross_divergence.md) | P=128, fractions 0.08, explicit percent denominator |
| 09 | [Multiple phases](calculations/task09_multicomponent.md) | 2%+6% overhead assumption |
| 10 | [Memory growth](calculations/task10_memory_growth.md) | 64/512-node capacity, missing-byte caveat |
| 11 | [Sun–Ni comparison](theory/task11_sun_ni_comparison.md) | Memory-bounded generalization |
| 12 | [Communication penalty](theory/task12_communication_penalty.md) | Correct useful-work denominator model |
| 13 | [Failure cases](theory/task13_failure_cases.md) | Algorithmic critical paths and contention |
| 14 | [Isoefficiency](theory/task14_isoefficiency.md) | Linear growth under stated ideal assumptions |
| 15 | [Energy scaling](theory/task15_energy_scaling.md) | Equal time vs equal work |
| 16 | [GEMM scaling](calculations/task16_gemm_scaling.md) | Compute work vs matrix memory |
| 17 | [Stencil scaling](calculations/task17_stencil_scaling.md) | Global vs local surface/volume |
| 18 | [N-body algorithms](analysis/task18_nbody_scaling.md) | All-pairs and Barnes–Hut dimension growth |
| 19 | [Monte Carlo](analysis/task19_monte_carlo.md) | PRNG, reductions, serialization |
| 20 | [Graph analytics](analysis/task20_graph_processing.md) | BFS/PageRank irregularity |
| 21 | [Simulator](benchmarks/simulator.py), [CSV](calculations/simulated_scaling.csv), [overlay PDF](plots/model_overlay.pdf) | Arbitrary parameters and analytical plots |
| 22 | [Benchmark](benchmarks/weak_scale_bench/benchmark.py), [instructions](benchmarks/weak_scale_bench/README.md) | Real process weak scaling |
| 23 | [Raw CSV](benchmarks/results.csv), [fitter](benchmarks/fit_gustafson.py), [report](analysis/task23_experimental_results.md) | 60 trials, mean/SD/SE, regression |
| 24 | [Audit](analysis/task24_hardware_audit.md), [vector PDF](plots/scaling_comparison.pdf), [PNG](plots/scaling_comparison.png) | Hardware, error bars and discrepancy |
| 25 | [This guide](README.md) | Synthesis, commands and artifact index |

## Supporting artifacts and provenance

- `benchmarks/models.py`: explicit model functions and fraction conversion.
- `benchmarks/results.metadata.json`: UUID, timings, host information, parameters,
  timing scope, software and completion state.
- `benchmarks/summary.csv`: means, sample standard deviations, standard errors,
  scaled efficiencies and distinct weak runtime efficiency.
- `benchmarks/fit.json`: fitted alpha, residuals, RMSE, R², standard-error caveat
  and unconstrained-line diagnostic.
- `benchmarks/report_results.py`: regenerate Task 23's report from validated data.
- `benchmarks/test_practice.py`: 15 automated tests, including real spawn workers,
  analytic identities, CSV validation, regression recovery and corrupt-input rejection.
- `benchmarks/verify_artifacts.py`: verifies all 25 task deliverables, Markdown
  local-link targets, simulation rows, image/PDF signatures, and source compilation.
- `benchmarks/logs/benchmark_stdout.txt`, `benchmark_stderr.txt`, `fit_stdout.txt`,
  `tests.txt`, `verification.json`: raw execution output and validation evidence.
- `plots/model_overlay.png` and `.pdf`: theoretical simulator overlays, labeled
  separately from empirical observations.
- `requirements-plots.txt`: optional plotting requirement.

Primary bibliographies are embedded in relevant reports. The historical
1024-processor results are literature findings, **not our laptop measurements**.
No `.pyc`, virtual environments, downloaded third-party code, or large input arrays
are committed. Source, formulas, CSV and reports are inspectable in Git. Atomic
task commits record the implementation; remote publication is a separate step.

## Engineering guidance

1. **Choose the question first.** Use Amdahl for latency of a fixed job and
   Gustafson for larger jobs at a time budget. Always name the serial reference.
2. **Scale useful work, not dimensions blindly.** GEMM, all-pairs, trees, stencils
   and graph traversals have different compute/memory growth and critical paths.
3. **Audit communication and decomposition.** Constant cells per rank can keep
   local halos constant but cannot make global reductions free. Poor partitions
   can increase traffic even when aggregate work grows linearly.
4. **Use memory bounds honestly.** Define bytes/cell, replicated state, halos,
   reserves and whether GB or GiB. A memory-capacity upper bound is not a timing
   guarantee or necessarily a realizable per-node layout.
5. **Measure real scope.** Keep startup-inclusive and steady-state results
   separately labeled. Compare identical useful serial/parallel work, retain
   outliers, repeat, test checksums, and export raw timings before fitting curves.
6. **Do not overinterpret alpha.** It is workload-, machine-, P- and timer-dependent.
   Model residuals reveal missed costs; a good line fit does not identify a cause.
7. **Budget energy as well as time.** A P-node fixed-time experiment usually uses
   more energy while solving more work. Compare matched work before claiming
   energy savings.
8. **Design for weak scaling when work is expandable and separable.** Monte Carlo
   batches, well-partitioned local-domain simulations and throughput ensembles are
   promising. Global synchronization, irregular frontiers and growing algorithmic
   complexity need explicit overhead/isoefficiency analysis before adding nodes.