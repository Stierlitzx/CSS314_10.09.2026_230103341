# Task 08 — Amdahl and Gustafson curve divergence

**Student ID:** 230103341

## My comparison

I use P for processor count, f for Amdahl's one-processor serial fraction (the handout's s), and α for Gustafson's non-scaling fraction of parallel elapsed time. I assume equal-speed processors, balanced work, and no extra overhead. The assignment sets both fractions numerically to 0.08 at P = 128, but their different references mean this is **not an equal-work performance comparison**.

```text
S_A = P/[1 + f(P − 1)]
S_G = P − α(P − 1)
Δ = S_G − S_A
D_A (%) = 100 × Δ/S_A
```

I use **Amdahl speedup as the discrepancy denominator**, not P, S_G, or their average. With the common numerical input x:

```text
f = α = x = 0.08 = 2/25; P = 128
S_A = 1/(0.08 + 0.92/128) = 3200/279 ≈ 11.469534050179211
S_G = 128 − 0.08 × 127 = 2946/25 = 117.84
Δ = 741934/6975 ≈ 106.370465949820789
D_A (%) = 100 × (741934/6975)/(3200/279)
        = 370967/400 = 927.4175
S_G/S_A = 410967/40000 = 10.274175
```

| P | Baseline s for Amdahl | Machine α for Gustafson | Exact S_A | Exact S_G | Exact Δ | Discrepancy D_A (%) |
|---:|---:|---:|---|---|---|---:|
| 128 | 0.08 | 0.08 | 3200/279 | 2946/25 | 741934/6975 | 927.4175 |

Thus the Gustafson number is 1027.4175% **of**, or 927.4175% **greater than**, the Amdahl number.

## My crossover and consistency checks

```text
For f = α = x:
Δ = x(1 − x)(P − 1)²/[1 + x(P − 1)]
D_A (%) = 100 × x(1 − x)(P − 1)²/P
```

I find no interior crossover: Δ is positive for P > 1 and 0 < x < 1. Equality occurs only at P = 1, x = 0, or x = 1. For fixed interior x, S_A approaches 1/x while S_G grows linearly.

For matched workloads, serial time s_time and parallel time t on P processors give:

```text
α = s_time/(s_time + t)
f = s_time/(s_time + Pt) = α/[α + P(1 − α)]
α = Pf/[1 + f(P − 1)]

α = 2/25 → f = 1/1473 → S_A = S_G = 117.84
f = 2/25 → α = 256/279 ≈ 0.917562724014337
         → S_A = S_G = 3200/279
```

I checked the results and identities with exact fractions. The apparent gap disappears after reference conversion; it does not show one law outperforming the other.