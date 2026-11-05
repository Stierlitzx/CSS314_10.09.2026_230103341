# Task 18 — N-body scaling: direct summation and Barnes–Hut

**Student ID:** 230103341

## My compute-budget model

I compare one step with N particles, P workers, baseline N₀ > 1 and scaled count Nₚ. Excluding overhead, force work F satisfies:

```text
F(Nₚ)/P = F(N₀)
```

This scales work, not memory. I fix accuracy settings and particle statistics; Barnes–Hut/direct comparisons need an error criterion.

## Particle-growth laws

With constant cost factor k, direct summation gives:

```text
F(N) = kN²  ⇒  Nₚ = N₀√P
Exact pairs: N(N − 1)/2
Exact scaling: Nₚ(Nₚ − 1) = PN₀(N₀ − 1)
```

The square-root law is the large-N approximation. At 64 workers, particle count grows eightfold, not 64-fold. Choosing Nₚ = PN₀ multiplies compute time by P. Particle data need only O(N) storage unless I retain a full pair matrix; distributed position/force exchanges cost extra.

Barnes–Hut rebuilds a spatial tree each step, with O(N log N) force calculation [BH86]. Assuming F(N) = kN ln N:

```text
K = N₀ ln N₀; Nₚ ln Nₚ = KP
Nₚ = exp(W₀(KP)) = KP/W₀(KP)
W₀(z) exp(W₀(z)) = z
As P → ∞: W₀(KP) ∼ ln(KP)
Nₚ ∼ KP/ln(KP) = Θ(P/ln P)
```

Here ln is natural logarithm and W₀ the principal Lambert function. For N₀ = 1024, P = 64, direct scaling gives 8192 particles; Barnes–Hut approximately 42613.86, with work ratio 64. The naive 65536 raises compute time by ln(65536)/ln(1024) = 1.6. These are within-algorithm growth factors, not equal runtimes; asymptotics are not finite-size solutions.

## Serial-work limitation

I define elapsed time Tₚ, serial setup/update A, communication C, imbalance I and serial fraction αₚ:

```text
Tₚ(N) = A(N) + F(N)/P + C(P,N) + I(P,N)
αₚ = A(Nₚ)/Tₚ(Nₚ)
```

Force complexity cannot determine αₚ. Constant A and negligible overhead preserve it. Illustratively, with serial A(N) = aN and constant force time B, A grows as aN₀√P or aKP/W₀(KP), respectively; αₚ = A/(A+B) → 1. Parallel tree construction, initialization and output can change this—not a measured property.

Fixed total time T* instead requires Tₚ(Nₚ) = T*. For a > 0 and nonnegative costs, Nₚ ≤ T*/a. Unbounded growth thus needs bounded/parallelized non-force phases. My validation plan separates force, tree, communication, integration and output, checks clustered-input imbalance and direct-reference errors, and fixes tolerance/time-step policy while reporting simulated physical time. I claim no empirical performance.

- [BH86] Barnes and Hut, “A hierarchical O(N log N) force-calculation algorithm,” Nature 324, 446–449 (1986): [publication/abstract](https://www.nature.com/articles/324446a0), [DOI](https://doi.org/10.1038/324446a0).
