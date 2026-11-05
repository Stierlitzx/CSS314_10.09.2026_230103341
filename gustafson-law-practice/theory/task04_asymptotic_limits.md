# Task 04 — Asymptotic speedup and efficiency

**Student ID:** 230103341

## 1. Specify the family before taking a limit

Let \(P\) tend to infinity through positive integers. The two standard idealized limits hold different quantities constant:

- **Amdahl family:** one workload is fixed, with single-core serial fraction \(s_A\).
- **Gustafson family:** workload grows, with parallel-observed serial fraction \(\alpha_P=\alpha\) fixed.

Both exclude additional communication, imbalance and redundant computation. Gustafson's original article motivates the second family by fixing elapsed time rather than problem size. [G88] Without a specified workload path and parameter convention, an asymptotic statement about “the serial fraction” is incomplete.

## 2. Fixed-work Amdahl speedup

For fixed \(s_A\in[0,1]\),

\[
S_A(P;s_A)=\frac{P}{1+(P-1)s_A}
=\frac{1}{s_A+(1-s_A)/P}.
\]

### Case A: \(0<s_A\leq1\)

Since \((1-s_A)/P\to0\),

\[
\boxed{\lim_{P\to\infty}S_A(P;s_A)=\frac1{s_A}}.
\]

For \(0<s_A<1\), the ceiling is approached from below, with expansion

\[
S_A(P;s_A)=\frac1{s_A}
-\frac{1-s_A}{P s_A^2}+O(P^{-2}).
\]

This expansion is for fixed positive \(s_A\); it is not uniform near \(s_A=0\). When \(s_A=1\), speedup is exactly 1 for all \(P\).

### Case B: \(s_A=0\)

There is no serial work, so

\[
\boxed{S_A(P;0)=P\longrightarrow\infty}.
\]

The expression \(1/s_A\) must not be used as a finite ceiling at zero. The direct formula resolves this boundary case.

## 3. Fixed-work Amdahl efficiency — including the zero-serial exception

\[
E_A(P;s_A)=\frac{S_A}{P}
=\frac1{1+(P-1)s_A}.
\]

Therefore

\[
\boxed{
\lim_{P\to\infty}E_A(P;s_A)=
\begin{cases}
0,&0<s_A\leq1,\\
1,&s_A=0.
\end{cases}}
\]

**The exception \(s_A=0\) is essential:** perfect parallelism gives efficiency exactly 1 at every finite processor count, and its limit is 1, not 0. For any fixed positive \(s_A\),

\[
E_A(P;s_A)\sim\frac1{P s_A}.
\]

The efficiency loss is not evidence that the serial section grows. The serial section sets a nonzero runtime floor while additional processors provide decreasing useful benefit.

For a target efficiency \(\eta\in(0,1]\) and \(s_A>0\),

\[
E_A\geq\eta
\iff
P\leq1+\frac{1/\eta-1}{s_A}.
\]

For example, \(s_A=0.01\) permits at most \(P=101\) at efficiency at least 50% in the ideal model. Extra overhead can only worsen this bound if it adds nonnegative elapsed time without changing the reference workload.

## 4. Fixed-fraction Gustafson speedup and efficiency

For fixed \(\alpha\in[0,1]\),

\[
S_G(P;\alpha)=P(1-\alpha)+\alpha,\qquad
E_G(P;\alpha)=1-\alpha+\frac{\alpha}{P}.
\]

If \(0\leq\alpha<1\),

\[
\boxed{S_G(P;\alpha)\longrightarrow\infty},\qquad
\boxed{\frac{S_G(P;\alpha)}{P}\longrightarrow1-\alpha}.
\]

The speedup is linear, with slope \(1-\alpha\). Its absolute shortfall from \(P\) is \((P-1)\alpha\), which grows even though efficiency tends to a positive constant. “Unbounded speedup” therefore does not mean “perfect efficiency.”

At the boundaries:

- \(\alpha=0\): \(S_G=P\), \(E_G=1\).
- \(\alpha=1\): \(S_G=1\), \(E_G=1/P\to0\); no parallelizable work is being added.

For \(\alpha>0\), efficiency decreases toward \(1-\alpha\) from above, because \(dE_G/dP=-\alpha/P^2\). This is a changing-workload comparison, not the fixed-work efficiency limit from §3.

## 5. Summary of all endpoint cases

| Family and held-fixed fraction | Speedup limit | Efficiency limit |
|---|---:|---:|
| Amdahl, \(s_A=0\) | \(+\infty\), exactly \(P\) | **1** |
| Amdahl, \(0<s_A<1\) | \(1/s_A\) | 0 |
| Amdahl, \(s_A=1\) | 1 | 0 |
| Gustafson, \(\alpha=0\) | \(+\infty\), exactly \(P\) | 1 |
| Gustafson, \(0<\alpha<1\) | \(+\infty\), slope \(1-\alpha\) | \(1-\alpha\) |
| Gustafson, \(\alpha=1\) | 1 | 0 |

## 6. Why these limits are compatible

For the same ideal decomposition at each \(P\),

\[
s_A=\frac{\alpha_P}{P-(P-1)\alpha_P},\qquad
\alpha_P=\frac{P s_A}{1+(P-1)s_A}.
\]

### Hold the workload and \(s_A>0\) fixed

Then \(\alpha_P\to1\), not a constant below one. In fact,

\[
1-\alpha_P=\frac{1-s_A}{1+(P-1)s_A},
\]

and inserting this changing \(\alpha_P\) into \(P-(P-1)\alpha_P\) recovers the finite Amdahl ceiling.

### Hold \(0<\alpha<1\) fixed along scaled workloads

Then the one-core fraction of each new workload decreases:

\[
s_A(W_P)\sim\frac{\alpha}{(1-\alpha)P}\to0.
\]

One cannot apply the fixed-positive-\(s_A\) limit to this sequence. Substitution into Amdahl's expression gives the same unbounded scaled speedup. The equality of the formulas at every finite \(P\) is preserved.

## 7. A joint-limit criterion

For workload-dependent fractions \(s_A(P)\),

\[
E(P)=\frac1{1+(P-1)s_A(P)}.
\]

Thus the product \((P-1)s_A(P)\), not merely \(s_A(P)\to0\), determines asymptotic efficiency:

\[
\begin{array}{c|c}
(P-1)s_A(P)\text{ behavior}&E(P)\text{ limit}\\ \hline
\longrightarrow0&1\\
\longrightarrow c\in(0,\infty)&1/(1+c)\\
\longrightarrow\infty&0
\end{array}
\]

For example, \(s_A(P)=P^{-1/2}\to0\) still gives efficiency tending to zero, although speedup grows approximately as \(\sqrt P\). A fixed Gustafson fraction corresponds to \(c=\alpha/(1-\alpha)\), producing \(1/(1+c)=1-\alpha\).

This also exposes noncommuting efficiency limits:

\[
\lim_{s_A\to0^+}\lim_{P\to\infty}E_A(P;s_A)=0,
\qquad
\lim_{P\to\infty}\lim_{s_A\to0^+}E_A(P;s_A)=1.
\]

At fixed positive serial fraction, increasing processors eventually exposes the bottleneck. Removing the serial fraction first produces perfect scaling instead.

## 8. Numerical checks and practical qualification

For fixed-work \(s_A=0.01\),

| \(P\) | \(S_A\) | \(E_A\) |
|---:|---:|---:|
| 10 | 9.174312 | 0.917431 |
| 100 | 50.251256 | 0.502513 |
| 1000 | 90.991811 | 0.090992 |

The speedup approaches 100, while efficiency tends to zero. For fixed parallel-observed \(\alpha=0.01\), the corresponding scaled speedups are 9.91, 99.01 and 990.01, with efficiencies 0.991, 0.9901 and 0.99001, approaching 0.99. These tables use different fraction conventions and different workload families.

If additional overhead is \(h(P,W_P)\), a nominal scaled family with one-core reference \(a+P b_0\) instead has

\[
E(P)=\frac{a/P+b_0}{a+b_0+h(P,W_P)}.
\]

Bounded convergent overhead \(h\to h_\infty\geq0\) yields efficiency \(b_0/(a+b_0+h_\infty)\); unbounded overhead yields efficiency zero. The fixed-\(\alpha\) ideal limit is therefore conditional. The original detailed study explicitly measures several efficiency losses that the two-component ideal model suppresses. [GMB88]

## Bibliography — consulted primary sources

- **[G88]** John L. Gustafson, “Reevaluating Amdahl's Law,” *Communications of the ACM* 31(5), 532–533 (1988). [Author-hosted article text](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [publisher metadata](https://doi.org/10.1145/42411.42415). The asymptotic and joint-limit proofs above are derived explicitly from the stated formulas.
- **[GMB88]** John L. Gustafson, Gary R. Montry and Robert E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM Journal on Scientific and Statistical Computing* 9(4), 609–638 (1988). [Author-hosted full text](http://www.johngustafson.net/pubs/pub16/1024.html), §§2.1–2.2 and §8; web presentation notes minor 1995 revisions.
