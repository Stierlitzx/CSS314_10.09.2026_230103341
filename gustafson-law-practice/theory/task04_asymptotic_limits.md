# Task 04 — Asymptotic speedup and efficiency

**Student ID:** 230103341

## Fixed-work and scaled-work limits

I let P → ∞ through positive integers, excluding communication, imbalance and redundant operations. I hold one-core serial fraction s_A fixed for Amdahl's workload, but parallel-observed fraction α fixed for Gustafson's growing workload. [G88]

```text
S_A = P/[1+(P−1)s_A]             E_A = 1/[1+(P−1)s_A]
S_G = P(1−α)+α                  E_G = 1−α+α/P
```

I obtain all endpoint cases:

| Fixed parameter | Speedup limit | Efficiency limit |
|---|---:|---:|
| s_A = 0 | ∞, exactly P | 1 |
| 0 < s_A < 1 | 1/s_A | 0 |
| s_A = 1 | 1 | 0 |
| α = 0 | ∞, exactly P | 1 |
| 0 < α < 1 | ∞, slope 1−α | 1−α |
| α = 1 | 1 | 0 |

The zero-serial Amdahl exception is essential. For fixed positive s_A, I derive:

```text
S_A = 1/s_A − (1−s_A)/(P s_A²) + O(P⁻²)
E_A ~ 1/(P s_A)
E_A ≥ η  ⇔  P ≤ 1+(1/η−1)/s_A       for 0 < η ≤ 1
```

The expansion is not uniform near s_A = 0. At s_A = 0.01, efficiency ≥ 50% requires P ≤ 101. For Gustafson, dE_G/dP = −α/P²; efficiency approaches 1−α from above, while the absolute speedup shortfall (P−1)α grows.

## Compatibility and joint limits

For the same ideal workload, I use:

```text
s_A = αₚ/[P−(P−1)αₚ]
αₚ = P s_A/[1+(P−1)s_A]
```

Fixed positive s_A makes αₚ → 1, with 1−αₚ = (1−s_A)/[1+(P−1)s_A]. Fixed 0 < α < 1 instead gives s_A ~ α/[(1−α)P] → 0. Thus the different limits do not contradict algebraic equivalence.

For variable s_A(P), I obtain E = 1/[1+(P−1)s_A(P)]. If the product (P−1)s_A(P) tends to 0, positive c, or ∞, efficiency tends to 1, 1/(1+c), or 0 respectively. For s_A = P⁻¹ᐟ², efficiency still vanishes and S ~ √P. Fixed α corresponds to c = α/(1−α). Taking P → ∞ before s_A → 0⁺ gives efficiency 0; reversing these limits gives 1.

## Numerical and overhead checks

| P | S_A, s_A = 0.01 | E_A | S_G, α = 0.01 | E_G |
|---:|---:|---:|---:|---:|
| 10 | 9.174312 | 0.917431 | 9.91 | 0.991 |
| 100 | 50.251256 | 0.502513 | 99.01 | 0.9901 |
| 1000 | 90.991811 | 0.090992 | 990.01 | 0.99001 |

With scaled reference a+P b₀ and added nonnegative overhead h, I obtain E = (a/P+b₀)/(a+b₀+h). If h → h∞, the limit is b₀/(a+b₀+h∞); unbounded h forces zero efficiency. The ideal limit is conditional. [GMB88]

## References

- **[G88]** Gustafson, “Reevaluating Amdahl's Law,” *CACM* 31(5), 532–533 (1988): [article](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [DOI](https://doi.org/10.1145/42411.42415).
- **[GMB88]** Gustafson, Montry and Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM J. Sci. Stat. Comput.* 9(4), 609–638 (1988): [§§2.1–2.2, §8](http://www.johngustafson.net/pubs/pub16/1024.html), minor 1995 web revisions.