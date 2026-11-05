# Task 19 — Monte Carlo: independent work, nonzero coordination

**Student ID:** 230103341

## My scaled-work model

I assume independent samples Xᵢ with mean μ and finite variance σ². Each of P workers evaluates m samples at cost k seconds/sample. For total M, setup A, elapsed times T, speedup S and efficiency E:

```text
μ̂ = (X₁ + ⋯ + X_M)/M; Var(μ̂) = σ²/M
M = Pm; worker compute time B = km
Tₚ = A + km; T₁(M) = A + kPm
S = (A + kPm)/(A + km)
```

Gustafson scaling yields more samples in nearly constant time. Standard error falls as P⁻¹ᐟ², not P⁻¹; halving it needs fourfold samples. Correlated Markov-chain samples need effective sample-size analysis.

## Coordination remains in elapsed time

I reduce counts, sums and second moments. For r-byte aggregates, stage latency τ and bandwidth b, my unoverlapped binary-tree model is:

```text
ℓ = τ + r/b
C_red(P) = ℓ log₂P                  [P a power of two]
S_useful = (A + kPm)/(A + km + ℓ log₂P + C_rng(P) + C_out(P))
E = S_useful/P
```

C_rng and C_out cover exposed stream setup and output. Other counts need approximately ceil(log₂P) stages; all-reduce may add dissemination. Communication is not numerator sample work. Fixed m, positive ℓ and negligible other overhead yield Θ(P/log P) speedup and vanishing efficiency despite independent samples.

For illustrative A = 0.01 s, km = 1 s, P = 1024 and ℓ = 10 microseconds, depth is 10 and reduction costs 0.0001 s:

```text
S = 1024.01/1.0101 ≈ 1013.771; E ≈ 0.99001
```

Finite-scale efficiency can remain excellent. Sequential collection exposes Θ(P) root work; saving samples expands output from O(1) aggregates to Θ(Pm).

## My correctness and measurement plan

Different rank seeds cannot guarantee independent, nonoverlapping streams. I would use documented skip-ahead/block splitting with bounded disjoint subsequences, allowing variable draws; or tested counter-based generation keyed by experiment, global sample ID, draw and block/lane. Mapping must be injective within supported ranges, without key/counter/lane reuse. Random123 provides stateless Nth-output access [R123]; avoiding reuse does not prove deterministic-generator independence. Application validation remains necessary.

Global sample IDs preserve a fixed sample collection across P. Floating-point reduction order can still change sums; I would document reproducible accumulation/a deterministic tree or justified tolerance, and check integer-count overflow for π estimation.

I would report samples/s, count, standard error/confidence interval, PRNG/mapping and setup/evaluation/reduction/output/total times. Fixed-work tests retain M; scaled speedup needs the enlarged Pm job’s serial time, measured or explicitly modeled—not the small job. Variable costs may need straggler-aware scheduling. These are analytical calculations, not measured Random123 performance.

- [R123] Salmon, Moraes, Dror and Shaw, “Parallel Random Numbers: As Easy as 1, 2, 3,” SC11 (2011): [DOI](https://doi.org/10.1145/2063384.2063405); [authors’ explanation and paper links](https://www.thesalmons.org/john/random123/).
