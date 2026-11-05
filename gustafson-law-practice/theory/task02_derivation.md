# Task 02 — Derivation of Gustafson's law and parameter conversion

**Student ID:** 230103341

## 1. Start with dimensional times, not ambiguous percentages

Fix a processor count \(P\geq1\) and a workload \(W_P\) chosen for that machine. Let

- \(t_s\): elapsed time of serial work, executed once;
- \(t_p\): elapsed time of the balanced parallel phase on \(P\) processors;
- \(T_P=t_s+t_p>0\): total parallel runtime.

Assume identical processors, unchanged per-operation execution rates, ideal load balance, and no extra parallel operations or overhead. All processors are assumed available throughout the parallel phase, so its one-core work equivalent is \(P t_p\), not \(t_p\). The corresponding one-core runtime for **the same workload** is

\[
T_1(W_P)=t_s+P t_p.
\]

This is the counterfactual reference used in scaled speedup. It is not the runtime of a smaller one-processor problem. The original paper uses parallel-system serial and parallel quantities \(s'\) and \(p'\) for this argument. [G88]

## 2. Normalize by the parallel execution time

Define the observed parallel-run serial fraction

\[
\alpha_P=\frac{t_s}{T_P},\qquad
1-\alpha_P=\frac{t_p}{T_P}.
\]

Then

\[
\begin{aligned}
S_G(P;\alpha_P)
&=\frac{T_1(W_P)}{T_P(W_P)}\\
&=\frac{t_s+P t_p}{t_s+t_p}\\
&=\alpha_P+P(1-\alpha_P)\\
&=\boxed{P-(P-1)\alpha_P}.
\end{aligned}
\]

Setting \(T_P=1\) is only a change of time units; it does not assert that the physical execution took one second. In those units, the parallel timeline consists of \(\alpha_P\) serial time units followed by \(1-\alpha_P\) parallel time units. Replacing the latter by its one-core equivalent produces \(\alpha_P+P(1-\alpha_P)\).

For an ideal fixed-time workload family, one further assumes \(\alpha_P=\alpha\), independent of \(P\). Only with that additional scaling assumption is the graph of speedup against processor count a straight line with constant slope \(1-\alpha\).

## 3. Efficiency and the cost of serial execution

Dividing the speedup by \(P\),

\[
\boxed{E_G(P;\alpha_P)=1-\alpha_P+\frac{\alpha_P}{P}}.
\]

Relative to ideal speedup \(P\),

\[
P-S_G=(P-1)\alpha_P.
\]

This loss can also be understood as idle processor capacity. During the serial phase, \(P-1\) processors cannot contribute useful work. The unused capacity is \((P-1)t_s\), out of the available \(P T_P\) processor-time units. Therefore

\[
1-E_G=\frac{(P-1)t_s}{P T_P}
=\left(1-\frac1P\right)\alpha_P.
\]

This capacity argument concerns the stated ideal model; busy processors doing redundant work need not deliver useful-work efficiency near one. [GMB88, §2.2.1]

## 4. Derive both parameter mappings

The Amdahl serial fraction is normalized by the one-core runtime of the chosen workload:

\[
s_A=\frac{t_s}{t_s+P t_p}.
\]

Substituting \(t_s=\alpha_P T_P\) and \(t_p=(1-\alpha_P)T_P\) gives

\[
\boxed{s_A=\frac{\alpha_P}{\alpha_P+P(1-\alpha_P)}
=\frac{\alpha_P}{P-(P-1)\alpha_P}}.
\]

Solve for \(\alpha_P\) without changing the workload:

\[
\begin{aligned}
s_A[P-(P-1)\alpha_P]&=\alpha_P,\\
P s_A&=\alpha_P[1+(P-1)s_A],\\
\alpha_P&=\boxed{\frac{P s_A}{1+(P-1)s_A}}.
\end{aligned}
\]

The denominators are positive for \(P\geq1\) and fractions in \([0,1]\). Thus the mappings include the boundary values without singularities. They are inverse transformations for a specified \(P\), not a claim that \(s_A\) and \(\alpha_P\) remain constant along the same workload family.

## 5. Algebraic equivalence with Amdahl's expression

Starting from the fixed-work expression and substituting the first mapping,

\[
\begin{aligned}
S_A
&=\frac{P}{1+(P-1)s_A}\\
&=\frac{P}{1+(P-1)\alpha_P/[P-(P-1)\alpha_P]}\\
&=P-(P-1)\alpha_P=S_G.
\end{aligned}
\]

Equivalently, using \(a=t_s\) and \(b=P t_p\) gives both formulas directly from \((a+b)/(a+b/P)\). Gustafson's contribution is the scaled-work interpretation and choice of normalization, not a failure of the fixed-work algebra.

## 6. Worked example and inverse check

Suppose an eight-processor execution takes 10 seconds: one second serial and nine seconds parallel. Then

\[
P=8,\quad \alpha_8=0.1,\quad
T_1=1+8(9)=73\text{ s}.
\]

Hence

\[
S_G=\frac{73}{10}=7.3,\qquad E_G=\frac{7.3}{8}=0.9125.
\]

The one-core fixed-work serial fraction is

\[
s_A=\frac1{73}=\frac{0.1}{8-7(0.1)}\approx0.01369863.
\]

The forward conversion returns

\[
\alpha_8=\frac{8/73}{1+7/73}=\frac8{80}=0.1,
\]

and Amdahl's speedup is \(1/[1/73+(72/73)/8]=7.3\). Using \(s_A=0.1\) instead would describe a different one-core workload decomposition.

## 7. Boundary and interpretation checks

- \(P=1\): \(S_G=E_G=1\), and \(s_A=\alpha_1\).
- \(\alpha_P=0\): no serial phase, \(S_G=P\), \(E_G=1\), \(s_A=0\).
- \(\alpha_P=1\): entirely serial, \(S_G=1\), \(E_G=1/P\), \(s_A=1\).
- \(0\leq\alpha_P\leq1\): \(1\leq S_G\leq P\), as required by this overhead-free model.
- For fixed \(P>1\), \(\partial S_G/\partial\alpha_P=-(P-1)\): a one-percentage-point rise in the parallel-observed serial fraction reduces predicted speedup by \(0.01(P-1)\).

If an experiment yields \(S>P\), these assumptions must be revisited; cache, memory hierarchy, reference algorithms or work counts may have changed. If only a measured “serial percentage” is supplied, first establish its timing denominator and which overheads it includes.

## Bibliography — consulted primary sources

- **[G88]** John L. Gustafson, “Reevaluating Amdahl's Law,” *Communications of the ACM* 31(5), 532–533 (1988). [Author-hosted article text](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [publisher metadata](https://doi.org/10.1145/42411.42415).
- **[GMB88]** John L. Gustafson, Gary R. Montry and Robert E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM Journal on Scientific and Statistical Computing* 9(4), 609–638 (1988). [Author-hosted full text, §§2.1–2.2.2](http://www.johngustafson.net/pubs/pub16/1024.html). This web presentation identifies minor revisions made in 1995.
