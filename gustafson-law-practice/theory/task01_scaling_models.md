# Task 01 — Fixed-work and scaled-work performance models

**Student ID:** 230103341

## 1. Quantities that must be distinguished

Let \(P\geq 1\) be the number of identical processors, and let \(W\) identify a workload, including its input, numerical resolution, algorithm and stopping criterion. Write

\[
T_1(W)=a(W)+b(W),\qquad
T_P(W)=a(W)+\frac{b(W)}{P}.
\]

Here \(a(W)\) is the time required for inherently serial work, and \(b(W)\) is the one-processor time for perfectly divisible parallelizable work. Assume positive total runtime. This ideal decomposition assumes unchanged processor performance, balanced work, no additional parallel operations and no communication or synchronization cost. It is a model, not an identity for arbitrary software.

The speedup and processor efficiency for the **same workload** are

\[
S(P,W)=\frac{T_1(W)}{T_P(W)},\qquad E(P,W)=\frac{S(P,W)}{P}.
\]

Two serial fractions use different denominators:

\[
\boxed{s_A(W)=\frac{a(W)}{a(W)+b(W)}}
\quad\text{(single-core, fixed-work reference)},
\]

\[
\boxed{\alpha_P(W)=\frac{a(W)}{a(W)+b(W)/P}}
\quad\text{(observed parallel-run serial fraction)}.
\]

The subscript in \(s_A\) identifies the Amdahl normalization, not a processor count. Unless the workload is specified, neither fraction should be described simply as “the serial percentage.” Gustafson's original article distinguishes its initial one-processor quantities from its later parallel-system quantities \(s',p'\); the notation here makes that distinction explicit. [G88]

## 2. Fixed work: Amdahl's model and strong scaling

Gustafson's 1988 article identifies Amdahl's fixed-work argument with Gene Amdahl's 1967 paper, “Validity of the single-processor approach to achieving large scale computing capabilities.” The historical attribution here follows the consulted 1988 primary article and its reference [1]. [G88]

Hold \(W=W_0\) fixed while changing \(P\). Consequently \(a,b,T_1\) and \(s_A\) remain fixed. Dividing both runtime components by \(T_1\) gives

\[
\boxed{S_A(P;s_A)=\frac{1}{s_A+(1-s_A)/P}
=\frac{P}{1+(P-1)s_A}}.
\]

The question is: **How much sooner can this particular problem finish?** This is the ideal strong-scaling model. The parallel part shrinks as \(1/P\), but the absolute serial time does not. For \(s_A>0\), this implies the speedup ceiling \(1/s_A\), not an unlimited linear improvement.

The parallel-observed fraction does not remain equal to \(s_A\):

\[
\alpha_P=\frac{P s_A}{1+(P-1)s_A}.
\]

Thus a program with a small single-core serial fraction can spend most of its eventual parallel elapsed time in its serial section.

## 3. Scaled work: Gustafson's model

Now consider a workload family \(W_P\). A useful ideal construction keeps the serial time \(a\) fixed and increases the one-processor parallel work from \(b_0\) to \(P b_0\):

\[
T_1(W_P)=a+P b_0,\qquad T_P(W_P)=a+b_0.
\]

The elapsed parallel runtime is constant, although the problem is larger. The observed parallel fraction is also constant along this family:

\[
\alpha=\frac{a}{a+b_0}.
\]

Comparing the parallel execution with a single processor solving **that same enlarged workload** yields

\[
\boxed{S_G(P;\alpha)=\frac{a+P b_0}{a+b_0}
=\alpha+P(1-\alpha)=P-(P-1)\alpha}.
\]

The question is: **How much larger a computation can be carried out within a given time budget?** This is Gustafson's scaled-speedup viewpoint. [G88]

It is related to weak scaling, but the terminology needs care. If parallel work per processor is fixed and serial time stays bounded, this construction has constant runtime. Merely fixing data items per processor does not guarantee constant computational work per processor: iteration counts, global reductions and numerical complexity can change with problem size. A common weak-scaling metric is \(T_1(W_1)/T_P(W_P)\); it equals 1 in this ideal construction, whereas Gustafson's processor efficiency is \(1-\alpha+\alpha/P\). These metrics use different numerators.

## 4. Exact relationship between the fractions

For the same \(W\) and \(P\), elementary normalization gives

\[
\boxed{\alpha_P=\frac{P s_A}{1+(P-1)s_A}},\qquad
\boxed{s_A=\frac{\alpha_P}{P-(P-1)\alpha_P}}.
\]

Substitution into either speedup expression gives the other. The laws are therefore compatible descriptions of the same ideal runtime decomposition. Their different scaling conclusions arise from what is held constant as \(P\) changes. It is invalid to insert an identical numerical “serial fraction” into both formulas without checking its denominator and workload.

| Property | Fixed-work family | Ideal fixed-time scaled family |
|---|---|---|
| Workload | \(W_0\) | \(W_P\) |
| One-core parallel work | \(b\), fixed | \(P b_0\) |
| Absolute serial time | \(a\), fixed | \(a\), fixed by assumption |
| One-core serial fraction | \(s_A\), fixed | \(a/(a+P b_0)\), decreasing |
| Parallel serial fraction | Increases with \(P\) if \(0<s_A<1\) | \(\alpha=a/(a+b_0)\), fixed |
| Main objective | Reduce time for one problem | Increase problem size within a time budget |

## 5. Numerical comparison with explicitly different inputs

For \(P=16\) and a **one-core fixed-work fraction** \(s_A=0.05\),

\[
S_A=\frac{16}{1+15(0.05)}=\frac{64}{7}\approx9.142857,
\qquad \alpha_{16}=\frac{16(0.05)}{1.75}\approx0.457143.
\]

Using the mapped fraction in Gustafson's expression gives \(16-15(0.457143)\approx9.142857\), as required.

For a different scaled workload with **parallel-observed fraction** \(\alpha_{16}=0.05\),

\[
S_G=16-15(0.05)=15.25,\qquad
s_A=\frac{0.05}{15.25}\approx0.003278689.
\]

The two results do not disagree: the second workload has a much smaller one-core serial fraction.

## 6. Applicability and limitations

In practice one should use a richer model such as

\[
T_P(W)=a(W)+\frac{b(W)}{P}+h(P,W),
\]

where \(h\) accounts for communication, synchronization, imbalance and additional operations, without double-counting costs already included in \(a\) or \(b\). With \(h>0\), a measured serial fraction alone is insufficient to determine speedup. The detailed Sandia study explicitly analyzes operation efficiency and hardware-induced imbalance rather than attributing every loss to an inherently serial algorithmic section. [GMB88]

## Bibliography — consulted primary sources

- **[G88]** John L. Gustafson, “Reevaluating Amdahl's Law,” *Communications of the ACM* 31(5), 532–533 (May 1988). [Author-hosted article text](http://www.johngustafson.net/pubs/pub13/amdahl.htm); [publisher record and DOI](https://doi.org/10.1145/42411.42415).
- **[GMB88]** John L. Gustafson, Gary R. Montry and Robert E. Benner, “Development of Parallel Methods for a 1024-Processor Hypercube,” *SIAM Journal on Scientific and Statistical Computing* 9(4), 609–638 (July 1988). [Author-hosted full text](http://www.johngustafson.net/pubs/pub16/1024.html), especially §§2–4. The HTML version states that minor revisions were made for its 1995 web presentation.
