# Task 22 — Empirical weak-scaling harness

**Student ID: 230103341.** Python multiprocessing, not Python threads: each worker has
its own interpreter and executes a CPU-bound midpoint integration of
`4/(1+x*x)` over disjoint points of `[0,1]`. The analytic answer is pi.
The benchmark and regression use only the standard library, consistent with the
existing CPU practice exercises. No compiler or NumPy is required.

## Run

The following PowerShell commands work from any directory on the measured host:

```powershell
$root = 'C:\Users\aiblb\PycharmProjects\CSS314_10.09.2026_230103341\gustafson-law-practice'
py "$root\benchmarks\weak_scale_bench\benchmark.py"
py "$root\benchmarks\fit_gustafson.py"
```

On another machine, set `$root` to its absolute checkout path; use `python3` in
place of `py` on Linux/macOS. Python **3.10+** is required. Plotting additionally
needs `py -m pip install -r "$root\requirements-plots.txt"`. `--no-plots` skips
that optional dependency in the fitting/simulation scripts.

Default settings: **k=1,000,000 evaluations per process**, every integer worker
count from 1 through `os.cpu_count()`, **five fresh trials** at each count.
No pool can silently reuse fewer workers: exactly P processes are explicitly
started, joined and checked by distinct PID in every trial. Logical processor
count is a hardware-concurrency limit, not a promise of P independent physical
cores or accounting for container quotas. User-selected counts must not exceed
the reported logical count.

Optional flags: `--workers 1,2,4,6,12 --k 1000000 --runs 5 --seed 230103341`.
Use `--output` with a new absolute CSV path for another experiment. Existing
datasets are protected unless `--overwrite` is explicitly supplied. A neighboring
`.metadata.json` records the session UUID, execution timestamps, host telemetry,
timing scope and completion/failure state; partial failed files must not be
reported as completed measurements.

## Work, timing and correctness

At P workers the total quadrature has **Pk points**, each worker handles exactly k.
Floating-point summation order is identical in direct sequential and parallel
versions (same chunks and `math.fsum` reduction). The serial reference executes
the **same scaled workload** on the parent CPU. We retain a serial reference
for every paired sample instead of assuming `Ts(Pk)=P*Ts(k)`.

The primary wall clock includes pipe allocation, process creation, Windows spawn,
numeric computation, result transfer, reduction, joining and cleanup. There is
no intentional sleep or adjustable synthetic serial fraction. Reference timings
exclude process overhead because they measure direct sequential execution.
Validation is outside both timers; each retained result must agree with the
serial reference and pi within the documented numerical tolerance. No input
array is allocated; memory use mainly reflects interpreter/process state.

Parent warmup precedes all samples. Configuration order is deterministically
shuffled in each repetition; the serial/parallel measurement order alternates.
This reduces systematic ordering bias but is not a substitute for frequency,
temperature, affinity or background-load control. All outliers remain in CSV.

## Two metrics, deliberately not conflated

- Scaled speedup: `Ts(Pk)/Tp(Pk)` (paired measured same-work serial reference).
- Scaled efficiency: `S(P)/P`.
- Weak runtime efficiency: `Tp(k)/Tp(Pk)` (one-worker spawned baseline divided
  by P-worker runtime). It need not equal scaled efficiency when reference
  costs/startup differ.

Task 23 fits the means of paired scaled speedups to `S=P-alpha(P-1)`.
Task 24 retains `P=1` even though direct-vs-spawn timing may give `S(1)<1`:
that residual documents a real failure of the ideal no-extra-overhead premise.
Fitted alpha is a **model-dependent effective overhead**, not measured serial code.