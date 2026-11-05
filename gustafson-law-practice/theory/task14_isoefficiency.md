# Task 14 — Isoefficiency and ideal weak scaling

**Student ID:** 230103341

## Ideal efficiency is only asymptotically constant

I take constant useful serial time A, per-worker parallel time B, and T = A+B > 0. With α = A/T, the scaled workload's one-core useful time is:

```text
W(P) = A+PB = T[α+P(1−α)]
E(P) = W/(PT) = 1−α+α/P
```

For 0 < α < 1, efficiency decreases from 1 toward 1−α, rather than remaining exactly constant. Work grows as Θ(P) only when B > 0. At α = 0, E = 1; at α = 1, work stays fixed and E = 1/P. This describes Gustafson's ideal family, not an arbitrary algorithm's isoefficiency. [G88]

## Formal overhead definition

For a specified algorithm, machine and input family, I define W using an appropriate sequential reference and aggregate processor-time overhead T_o:

```text
T_o(W,P) = P Tₚ(W)−W
E = W/(W+T_o)
E = E₀  ⇔  W = [E₀/(1−E₀)]T_o(W,P), 0 < E₀ < 1
```

For E ≥ E₀, I replace equality with ≥, assuming nonnegative overhead. T_o includes serial-phase idle capacity, communication, redundant work and imbalance—not merely elapsed communication time. Its W dependence can make the equation implicit. Added work must be useful and memory-feasible. [KG91]

## Three models

1. **Perfect division:** Tₚ = W/P gives T_o = 0 and E = 1. There is no nontrivial overhead-driven size requirement beyond granularity and available tasks; no positive workload realizes an exact lower E₀.
2. **Fixed serial work:** Tₚ = A+(W−A)/P, A > 0, gives T_o = (P−1)A. Thus W = [E₀/(1−E₀)](P−1)A = Θ(P). This is linear isoefficiency for this model, not exact constant efficiency along W = A+PB. Feasibility requires W ≥ A; P = 1 has E = 1.
3. **Exposed communication:** Adding elapsed C(P) gives T_o = (P−1)A+PC(P). For positive fixed coefficients and C independent of W, I obtain:

| C(P) | Required W at fixed E₀ |
|---|---|
| c log P | Θ(P log P) |
| d√P | Θ(P³ᐟ²) |
| c, constant | Θ(P) |

These require matching Θ growth, not only O upper bounds. Constant work per worker cannot absorb divergent communication. Since Tₚ = W/(PE₀), maintaining efficiency in the first two rows sacrifices fixed walltime: runtime grows as log P or √P.

## Workload-dependent serial cost

If A(W) = aW^γ, with a > 0 and 0 < γ < 1, negligible other costs give T_o ~ aP W^γ. I derive:

```text
W^(1−γ) = Θ(P)
W = Θ(P^[1/(1−γ)])
```

Linear isoefficiency disappears. A fixed positive intrinsically serial fraction prevents positive efficiency at unbounded P, regardless of input growth. Data-dependent communication adds further constraints.

## References

- **[G88]** Gustafson, scaled-work argument: [DOI](https://doi.org/10.1145/42411.42415).
- **[KG91]** V. Kumar and A. Gupta, “Analysis of scalability of parallel algorithms and architectures: a survey,” *ICS* (1991), 396–405: [DOI](https://doi.org/10.1145/109025.109118); [Crossref metadata](https://api.crossref.org/works/10.1145/109025.109118). I derive the equations for the stated models.