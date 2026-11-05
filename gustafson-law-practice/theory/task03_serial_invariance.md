# Task 03 — What does it mean for the serial part to remain invariant?

**Student ID:** 230103341

## 1. Three distinct invariance statements

“The serial part is constant” can refer to three different quantities:

1. **Absolute serial time:** \(a(W_P)\), measured in seconds.
2. **Single-core fixed-work serial fraction:**
   \[
   s_A(W_P)=\frac{a(W_P)}{a(W_P)+b(W_P)}.
   \]
3. **Parallel-observed serial fraction:**
   \[
   \alpha_P(W_P)=\frac{a(W_P)}{a(W_P)+b(W_P)/P}.
   \]

The two fractions have different denominators. Constancy of absolute serial time alone does not imply constancy of either fraction. Throughout this report, \(b(W_P)\) is the one-core time of parallelizable work, and the ideal runtime is \(a(W_P)+b(W_P)/P\). Extra parallel overhead is excluded unless explicitly introduced.

Gustafson describes unchanged serial costs as a **first approximation**, citing vector startup, program loading, serial bottlenecks and I/O, while the parallel work grows with the problem. That is a modeling assumption about selected applications and workload scaling, not a universal theorem about all input/output or serial algorithms. [G88]

## 2. Fixed work: constant serial seconds, changing observed fraction

For a fixed workload \(W_0\), write \(a=a(W_0)\), \(b=b(W_0)\). Both times and \(s_A=a/(a+b)\) remain fixed as \(P\) changes. Nevertheless,

\[
\alpha_P=\frac{a}{a+b/P}=\frac{P s_A}{1+(P-1)s_A}.
\]

Treating \(P\) as a continuous variable to establish monotonicity,

\[
\frac{d\alpha_P}{dP}
=\frac{s_A(1-s_A)}{[1+(P-1)s_A]^2}.
\]

Thus \(\alpha_P\) increases strictly for \(0<s_A<1\); the corresponding integer-processor sequence increases too. For \(s_A>0\), \(\alpha_P\to1\). The serial section has not become longer: the parallel section has become shorter. When \(s_A=0\), \(\alpha_P=0\) for every finite \(P\); when \(s_A=1\), \(\alpha_P=1\).

## 3. Gustafson's ideal scaled family: invariant serial seconds and parallel fraction

Let the workload increase so that

\[
a(W_P)=a,\qquad b(W_P)=P b_0,
\qquad a+b_0>0.
\]

Then

\[
T_P(W_P)=a+b_0,\qquad
\alpha_P(W_P)=\frac{a}{a+b_0}=\alpha,
\]

so both absolute serial time and the parallel-observed fraction stay fixed. However,

\[
s_A(W_P)=\frac{a}{a+P b_0}
=\boxed{\frac{\alpha}{P-(P-1)\alpha}}.
\]

For \(a>0\), \(b_0>0\),

\[
\frac{d s_A(W_P)}{dP}=-\frac{a b_0}{(a+P b_0)^2}<0,
\qquad s_A(W_P)\longrightarrow0.
\]

More precisely, for fixed \(0<\alpha<1\),

\[
s_A(W_P)\sim\frac{\alpha}{(1-\alpha)P}.
\]

This shrinking one-core fraction explains how a fixed absolute serial cost can coexist with growing scaled speedup. The single-core reference is processing ever more parallelizable work; it is not the same fixed workload used for an Amdahl limit.

## 4. A necessary and sufficient ratio condition

For positive serial time, allow \(a_P=a(W_P)\) and \(b_P=b(W_P)\) to vary arbitrarily. Then

\[
\alpha_P=\frac1{1+b_P/(P a_P)}.
\]

Therefore a constant \(\alpha_P=\alpha\in(0,1)\) is equivalent to

\[
\boxed{\frac{b_P}{P a_P}=\frac{1-\alpha}{\alpha}}
\quad\text{or}\quad
\boxed{b_P=P a_P\frac{1-\alpha}{\alpha}}.
\]

Constant parallel-observed fraction is thus a condition on a **ratio**, not necessarily on absolute seconds. If \(a_P\) doubles while \(b_P/P\) doubles, \(\alpha_P\) is unchanged but the total parallel time doubles. To conclude fixed runtime from a fixed fraction, one also needs fixed absolute serial time (or an equivalent total-time constraint).

The degenerate cases should be handled separately: \(\alpha=0\) means no serial time, and \(\alpha=1\) means no useful parallel phase in this decomposition.

## 5. Numerical experiment: the same baseline, two scaling paths

Start with \(a=2\) seconds and one-core parallel work \(b_0=18\) seconds. On one processor the serial fraction is \(2/20=0.1\).

### A. Keep the workload fixed

Use \(b=18\) at every processor count.

| \(P\) | \(a\) (s) | \(b/P\) (s) | \(T_P\) (s) | \(s_A\) | \(\alpha_P\) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 18 | 20 | 0.1 | 0.1 | 1 |
| 4 | 2 | 4.5 | 6.5 | 0.1 | 0.307692 | 3.076923 |
| 16 | 2 | 1.125 | 3.125 | 0.1 | 0.64 | 6.4 |

### B. Scale parallel work with the processor count

Use \(b_P=18P\), so the parallel phase always lasts 18 seconds.

| \(P\) | \(a\) (s) | \(b_P\) (one-core s) | \(T_P\) (s) | \(s_A(W_P)\) | \(\alpha_P\) | Scaled speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 18 | 20 | 0.1 | 0.1 | 1 |
| 4 | 2 | 72 | 20 | 0.027027 | 0.1 | 3.7 |
| 16 | 2 | 288 | 20 | 0.006897 | 0.1 | 14.5 |

The identical two-second serial section produces different fractions because the denominators evolve differently. The speedups answer different workload-scaling questions, not competing predictions for one experiment.

## 6. When the invariance assumption fails

Suppose parallel work grows linearly but serial work also increases:

\[
a_P=a_0 P^\gamma,\qquad b_P=P b_0,
\qquad a_0,b_0>0.
\]

Then

\[
T_P=a_0P^\gamma+b_0,\qquad
\alpha_P=\frac{a_0P^\gamma}{a_0P^\gamma+b_0}.
\]

- If \(\gamma=0\), the ideal constant-time, constant-\(\alpha\) construction is recovered.
- If \(\gamma>0\), runtime grows and \(\alpha_P\to1\); the constant-\(\alpha\) linear prediction is not justified.
- If \(\gamma<0\), serial work is being reduced as the machine grows, an additional algorithmic improvement rather than the usual serial-invariance assumption.

Increasing output volume, sequential preprocessing proportional to input size, and global control work can invalidate fixed \(a\). Communication and redundant operations can also grow without constituting literal single-processor serial code. The Sandia paper analyzes these losses separately and defines operation efficiency to account for extra parallel computation. [GMB88]

For empirical work, record at each \(P\): the precise workload, total elapsed time, serial time, communication/idle time, and the construction of the one-core reference. Only then decide which quantities can reasonably be held invariant.

## Conclusion

The serial *time* can stay constant while the one-core serial *fraction* decreases. In Gustafson's ideal scaled family, the parallel-observed fraction is constant because both serial time and per-processor parallel time are constant. In a fixed-work family, the same unchanged serial time occupies an increasing fraction of a shrinking parallel runtime. Denominator and workload labels resolve the ambiguity.

## Bibliography — consulted primary sources

- **[G88]** John L. Gustafson, “Reevaluating Amdahl's Law,” *Communications of the ACM* 31(5), 532–533 (1988). [Author-hosted original article text](http://www.johngustafson.net/pubs/pub13/amdahl.htm), particularly the first-approximation discussion of serial costs and the scaled-work derivation; [publisher record](https://doi.org/10.1145/42411.42415).
- **[GMB88]** John L. Gustafson, Gary R. Montry and Robert E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM Journal on Scientific and Statistical Computing* 9(4), 609–638 (1988). [Author-hosted full text](http://www.johngustafson.net/pubs/pub16/1024.html), §§2.1–2.2.2 and §8. The HTML presentation notes minor revisions in 1995.
