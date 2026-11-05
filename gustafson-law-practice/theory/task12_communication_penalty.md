# Task 12 — Communication as an elapsed-time penalty

**Student ID:** 230103341

## Useful work and exposed overhead

I take constant serial useful time A ≥ 0 and balanced parallel useful time B > 0 per worker. The scaled workload has ideal elapsed time T₀ = A+B and one-core useful time A+PB. Let C(P) ≥ 0 be exposed communication/synchronization time on the critical path, absent from the serial reference.

```text
Tₚ = A+B+C(P)
S = (A+PB)/[A+B+C(P)]
α = A/T₀; c(P) = C(P)/T₀
S = [α+P(1−α)]/[1+c(P)]
E = [1−α+α/P]/[1+c(P)]
```

I put overhead only in the denominator. C includes only unhidden communication; otherwise overlap is double-counted. The fraction α uses ideal useful time, not measured Tₚ. Applying Gustafson directly to A/Tₚ would wrongly reconstruct communication as useful computation. [G88]

## Growth models

I require positive coefficients and equalities or matching Θ growth, meaning matching asymptotic upper and lower bounds. An O(log P) upper bound alone also permits bounded or zero cost.

```text
C(P) = c log₂P, c > 0, P a power of two:
S = Θ(P/log P); E = Θ(1/log P) → 0

C(P) = d(√P−1), d > 0:
S = Θ(√P); E = Θ(P⁻¹ᐟ²) → 0
```

A fixed-fan-in tree with fixed-size messages motivates the first model; topology-dependent exchange may motivate the second. Neither is universal. The subtraction in √P−1 merely makes C(1) = 0.

For constant A,B, any C → ∞ forces E → 0. If C → C∞, then E → B/(A+B+C∞). For C = Θ(P), speedup stays bounded; superlinear communication can make S → 0.

## Numerical example and budget

At A = 1 s, B = 9 s and P = 64, I obtain 577 s one-core useful time and 10 s ideal parallel time. These are calculations, not measurements.

| Cost model | C(64), s | T₆₄, s | S | E |
|---|---:|---:|---:|---:|
| None | 0 | 10 | 57.700 | 0.90156 |
| 0.1 log₂P s | 0.6 | 10.6 | 54.434 | 0.85053 |
| 0.1(√P−1) s | 0.7 | 10.7 | 53.925 | 0.84258 |

For target E₀ > 0, I derive:

```text
E ≥ E₀  ⇔  C(P) ≤ (B+A/P)/E₀−(A+B)
```

A negative budget makes the target impossible even without communication. Its asymptotic budget is constant, so divergent C eventually violates every positive target.

For empirical comparison I would separate compute, collectives, imbalance and message volume without summing overlapping intervals, and preserve matched useful results. Increasing B may amortize communication, but changes the workload family and abandons this constant-work-per-worker premise.

## Reference

- **[G88]** Gustafson, “Reevaluating Amdahl's Law” (1988): [DOI](https://doi.org/10.1145/42411.42415). I derive the communication extensions here; they are not reported network measurements.