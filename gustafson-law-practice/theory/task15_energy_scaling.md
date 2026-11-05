# Task 15 — Energy scaling requires matched useful work

**Student ID:** 230103341

## Power and workload definitions

I use P for identical nodes, p_i ≥ 0 for baseline watts/node, and p_d ≥ 0 for **incremental** dynamic watts while computing. Total active power is p_i+p_d. If p_a already means total active power, I use p_d = p_a−p_i to avoid double-counting. Energy is J = ∫p(t) dt.

I assume unchanged node speed and constant phase power, initially excluding network, cooling and shared infrastructure. These are analytical models, not measurements.

For constant serial phase A and parallel phase B per worker, I define:

```text
T = A+B; α = A/T
S = α+P(1−α); E = S/P
T₁(Wₚ) = A+PB = ST
```

The small job W₁ takes T. If every allocated node draws full active power throughout, J₁(W₁) = (p_i+p_d)T and Jₚ(Wₚ) = P(p_i+p_d)T. Their ratio P compares different workloads. Relative useful work per joule is S/P = E, not S.

## Matched expanded job

For the same expanded workload, I obtain:

```text
J₁(Wₚ) = (p_i+p_d)ST
Full-active model: Jₚ/J₁ = P/S = 1/E ≥ 1
```

This is an upper power model for idle workers, not useful computation on every node during A. In a phase-aware model, all nodes retain baseline power, one computes during A, and all compute during B:

```text
Jₚ = P p_i T+p_d(A+PB) = T(P p_i+S p_d)
R_J = Jₚ/J₁ = [(P/S)p_i+p_d]/(p_i+p_d)
1 ≤ R_J ≤ P/S       for p_i+p_d > 0 and 1 ≤ S ≤ P
```

Dynamic energy matches serial dynamic energy; excess comes from baseline draw. If p_i = 0 or S = P, R_J = 1. Switching unused nodes off could remove idle baseline energy, but needs explicit transition-cost and availability assumptions.

## Numerical check and saving criterion

For P = 16, α = 0.1, T = 10 s, p_i = 30 W and p_d = 70 W, I obtain S = 14.5, E = 0.90625 and matched serial runtime 145 s.

| Execution | Energy |
|---|---:|
| Small job, one node, 10 s | 1,000 J |
| Expanded job, one node, 145 s | 14,500 J |
| Expanded job, 16 fully active nodes | 16,000 J |
| Expanded job, phase-aware power | 14,950 J |

Matched ratios are 1.10345 and 1.03103, not 16. For actual matched-work speedup S and average total-system powers p̄ₚ,p̄₁:

```text
Jₚ/J₁ = (p̄ₚ/p̄₁)/S
Jₚ ≤ J₁  ⇔  p̄ₚ/p̄₁ ≤ S
```

Thus energy savings require power growth no greater than speedup. Lower voltage, sleeping nodes or shared baseline amortization may help; communication and busy waiting may hurt. I would report the power boundary and joules per matched validated result.

## Reference

I use [Gustafson's scaled-work interpretation (1988)](https://doi.org/10.1145/42411.42415); the power equations and examples are derived models, not an empirical energy law.