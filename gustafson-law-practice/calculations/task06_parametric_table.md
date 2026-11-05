# Task 06 — Parametric Gustafson speedup and efficiency table

**Student ID:** 230103341

## Definitions and assumptions

Let `P` be the processor count and `α` the fraction of the **P-processor elapsed runtime** assigned to non-scaling execution. Normalize that runtime to one unit. Under a two-component model with balanced parallel work, equal-speed processors, no extra communication or scheduling overhead, and the same computational work when forming the serial reference, the elapsed components are `α` and `1−α`.

Executing that scaled workload serially would take `α+P(1−α)` units, so

\[
S_G(P,\alpha)=\frac{\alpha+P(1-\alpha)}1
=P-\alpha(P-1),
\]
\[
E_G(P,\alpha)=\frac{S_G}P
=1-\alpha+\frac{\alpha}P.
\]

`α` is not automatically the serial fraction of a fixed one-processor baseline. For the same two-component workload that baseline fraction would be

\[
f=\frac{\alpha}{\alpha+P(1-\alpha)}=\frac{\alpha}{S_G}.
\]

Assume the assignment's `α=0.04` remains applicable at each specified `P`; real workloads need not preserve it. This is a model table, not a benchmark.

## Parameter set and worked substitution

The handout specifies `α=0.04=1/25` and `P ∈ {16,64,256,1024,4096,16384}`. Hence

\[
S_G=\frac1{25}+\frac{24P}{25}=\frac{24P+1}{25},
\qquad E_G=\frac{24P+1}{25P}=0.96+\frac{0.04}{P}.
\]

For `P=1024`, substitute:

\[
S_G=1024-\frac1{25}(1024-1)=\frac{24577}{25}=983.08,
\]
\[
E_G=\frac{24577/25}{1024}=\frac{24577}{25600}
=0.9600390625=96.00390625\%.
\]

## Exact speedups and decimal efficiency

Every speedup decimal below terminates and is exact; rational forms make the arithmetic explicit. Efficiencies are shown as fractions and exact terminating percentages.

| P | α | Exact S_G | S_G decimal | Exact E_G | Efficiency (%) |
|---:|---:|---|---:|---|---:|
| 16 | 0.04 | 77/5 | 15.40 | 77/80 | 96.25 |
| 64 | 0.04 | 1537/25 | 61.48 | 1537/1600 | 96.0625 |
| 256 | 0.04 | 1229/5 | 245.80 | 1229/1280 | 96.015625 |
| 1024 | 0.04 | 24577/25 | 983.08 | 24577/25600 | 96.00390625 |
| 4096 | 0.04 | 19661/5 | 3932.20 | 19661/20480 | 96.0009765625 |
| 16384 | 0.04 | 393217/25 | 15728.68 | 393217/409600 | 96.000244140625 |

## Interpretation and checks

For fixed `α`, `S_G` is linear in `P`, with slope `1−α` and intercept `α`. The loss from ideal speedup is `P−S_G=α(P−1)`. Efficiency decreases toward `1−α` as `P` grows:

\[
\lim_{P\to\infty}E_G=1-\alpha=0.96=96\%.
\]

Thus nearly linear scaled speedup does not mean exactly 100% efficiency. At `P=16384`, the loss from ideal speedup is `0.04×16383=655.32`, matching `16384−15728.68=655.32`. The speedup has slope 0.96 and intercept 0.04; its efficiency approaches 96% from above.

These are scaled-work comparisons. `P` processors do not by themselves imply that a linear problem dimension should be multiplied by `P`: if useful work is proportional to `N^d`, constant per-processor work typically requires `N∝P^(1/d)`.

## Arithmetic validation log

Python `fractions.Fraction` evaluated the six assignment rows without intermediate floating-point rounding and checked `S_G=α+P(1−α)`, `E_G=S_G/P`, and `1≤S_G≤P`.

```text
alpha=1/25
P=16:    S_G=77/5=15.40;        E_G=77/80=.9625
P=64:    S_G=1537/25=61.48;     E_G=1537/1600=.960625
P=256:   S_G=1229/5=245.80;     E_G=1229/1280=.96015625
P=1024:  S_G=24577/25=983.08;   E_G=24577/25600=.9600390625
P=4096:  S_G=19661/5=3932.20;   E_G=19661/20480=.960009765625
P=16384: S_G=393217/25=15728.68;E_G=393217/409600=.96000244140625
6 exact rational assignment rows checked. PASS
```