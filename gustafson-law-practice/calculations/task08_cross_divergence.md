# Task 08 — Numerical divergence of Amdahl and Gustafson curves

**Student ID:** 230103341

## Definitions and comparison protocol

Amdahl's fixed-work speedup uses `f` (the handout's baseline `s`), the serial fraction of the **one-processor reference execution**. Here `f=s=0.08` and `P=128`:

\[
S_A(P;f)=\frac1{f+(1-f)/P}=\frac{P}{1+f(P-1)}.
\]

Gustafson's scaled-work speedup uses `α`, the non-scaling fraction of the **P-processor elapsed execution time**. The assignment requests the same numerical input `α=0.08` at `P=128`:

\[
S_G(P;\alpha)=\alpha+P(1-\alpha)=P-\alpha(P-1).
\]

Assume ideal balanced parallel work, identical processor rates, no unmodeled overhead, `P≥1`, and fractions between zero and one. Inserting the **same number** `x` into `f` and `α` is a comparison of two parameterized curves with **different reference fractions**. It is **not a true equal-work comparison or evidence that one law physically outperforms the other**. The models answer different scaling questions.

For a matched workload in the ideal two-component model, let `s` and `t` be serial and parallel elapsed times on the P-processor machine. Then

\[
\alpha=\frac{s}{s+t},\quad
f=\frac{s}{s+Pt}
=\frac{\alpha}{\alpha+P(1-\alpha)}.
\]

Using this converted `f` in Amdahl's expression yields exactly `S_A(P;f)=S_G(P;α)`. Conversely, `α=Pf/[1+f(P−1)]`. Equal symbols or equal numerical fractions without that conversion do not imply equal workloads.

## Discrepancy definition — Amdahl denominator explicitly

Define the signed difference and the percentage excess of the Gustafson number over the Amdahl number as

\[
\Delta=S_G-S_A,\qquad
\boxed{D_A=100\frac{S_G-S_A}{S_A}\%}.
\]

**The percentage denominator is Amdahl's speedup `S_A`, not Gustafson's speedup, P, or their average.** For these admissible same-number inputs, `Δ≥0`, so the absolute relative discrepancy has the same value. A symmetric percent, if desired, is `100|S_G−S_A|/[(S_G+S_A)/2]`; it is a different metric and is not used in the table.

With `f=α=x`, algebra gives

\[
\Delta
=\frac{x(1-x)(P-1)^2}{1+x(P-1)},\qquad
D_A=100\frac{x(1-x)(P-1)^2}{P}\%.
\]

## Assignment calculation: baseline s = 0.08 versus α = 0.08 at P = 128

Using the handout inputs `x=0.08=2/25`, first evaluate Amdahl:

\[
S_A=\frac1{0.08+0.92/128}
=\frac{128}{1+(2/25)(127)}
=\boxed{\frac{3200}{279}}
=11.469534050179211\ldots.
\]

For Gustafson, the same numerical fraction has the different elapsed-time reference:

\[
S_G=128-0.08(127)
=128-10.16=\boxed{117.84=\frac{2946}{25}}.
\]

The exact difference is

\[
\Delta=\frac{2946}{25}-\frac{3200}{279}
=\frac{741934}{6975}
=106.370465949820789\ldots.
\]

Using **Amdahl as the denominator**, as required:

\[
D_A=100\frac{2946/25-3200/279}{3200/279}\%
=\boxed{\frac{370967}{400}\%=927.4175\%}.
\]

Equivalently, `S_G/S_A=410967/40000=10.274175`: the Gustafson number is 1027.4175% **of** the Amdahl number, hence 927.4175% **greater than** it. Those two percentage statements must not be confused.

| P | Baseline s for Amdahl | Machine α for Gustafson | Exact S_A | Exact S_G | Exact Δ | Discrepancy D_A (%) |
|---:|---:|---:|---|---|---|---:|
| 128 | 0.08 | 0.08 | 3200/279 | 2946/25 | 741934/6975 | 927.4175 |

## No genuine crossover in the ideal same-number curves

For `P>1` and `0<x<1`, every factor in the numerator of `Δ` is positive and the denominator is positive. Therefore `S_G>S_A` throughout: there is **no interior crossover**. Equality occurs only in the degenerate cases:

* `P=1`: both speedups are 1;
* `x=0`: both predict `P`;
* `x=1`: both predict 1.

For fixed `0<x<1`, `S_A→1/x`, whereas `S_G=(1−x)P+x` grows linearly. Their discrepancy grows; this is a consequence of the distinct reference scenarios, not a contradiction between laws. A real timing model with differing workloads or `P`-dependent overhead can have other intersections, but that is not the constant-parameter model analyzed here.

At the assignment point `P=128`, conversion from `α=0.08` for a matched workload gives

\[
f=\frac{2/25}{2946/25}=\frac1{1473},\quad
S_A(128;1/1473)
=\frac{128}{1+127/1473}=\frac{2946}{25}=117.84.
\]

Conversely, keeping the assignment's baseline `s=f=0.08` and converting to the P-processor reference yields `α=128(2/25)/[1+(2/25)(127)]=256/279≈0.917562724014337`, not 0.08. With that converted `α`, Gustafson gives `3200/279`, matching Amdahl. The equality after conversion makes the reference-fraction issue concrete; the assignment's same-number discrepancy is not a genuine equal-work performance gap.

## Arithmetic validation log

Python `fractions.Fraction` checked the assignment values and both algebraic discrepancy identities exactly. Decimal displays are rounded only after the rational calculations.

```text
x=2/25, P=128
S_A=3200/279=11.469534050179211...
S_G=2946/25=117.84
delta=741934/6975=106.370465949820789...
D_A=100*delta/S_A=370967/400 %=927.4175 %
S_G/S_A=410967/40000=10.274175
alpha=2/25 -> f=1/1473 -> S_A=S_G=2946/25
f=2/25 -> alpha=256/279 -> S_A=S_G=3200/279. PASS
```