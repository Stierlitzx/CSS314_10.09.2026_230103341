# Task 14 — Isoefficiency and the limits of ideal weak scaling

**Student ID:** 230103341

## 1. The exact ideal efficiency is not exactly constant

For an ideal Gustafson family, take serial useful time \(A\), parallel useful time per worker \(B\), and \(T=A+B\), all independent of \(P\). Let \(\alpha=A/T\). The useful serial work of the scaled problem, expressed in single-processor seconds, is

\[
W(P)=A+PB=T[\alpha+P(1-\alpha)].
\]

Consequently,

\[
\boxed{E(P)=\frac{W}{PT}=1-\alpha+\frac{\alpha}{P}}.
\]

For \(0<\alpha<1\), efficiency decreases from 1 at \(P=1\) toward \(1-\alpha\); it is **near constant at large \(P\)**, not identically constant. The useful-work growth is \(W=\Theta(P)\) when \(B>0\). At \(\alpha=0\), \(E=1\); at \(\alpha=1\), work does not grow and \(E=1/P\). Thus a claim of linear work growth must exclude the all-serial endpoint.

This workload path is consistent with Gustafson's ideal fixed-time reconstruction. [G88] It does not establish the isoefficiency of an arbitrary algorithm whose serial work and communication depend on input size.

## 2. Formal definition through processor-time overhead

For a specified algorithm, machine and family of problems, let \(W\) be the time of a chosen appropriate sequential reference algorithm. Define

\[
T_o(W,P)=P T_P(W)-W,
\qquad
E(W,P)=\frac{W}{W+T_o(W,P)}.
\]

This \(T_o\) is an aggregate **processor-time** cost, not necessarily a single elapsed communication interval. It includes unused processor capacity during serial work, redundant computation, communication and imbalance. The classical isoefficiency framework asks how rapidly \(W\) must grow with \(P\) to maintain a chosen target \(0<E_0<1\). [KG91]

At exact target efficiency,

\[
\boxed{W=\frac{E_0}{1-E_0}T_o(W,P)}.
\]

For efficiency at least \(E_0\), replace equality by \(W\geq E_0T_o/(1-E_0)\), assuming nonnegative overhead. Since \(T_o\) may itself depend on \(W\), this is often an implicit equation, not a prescription to choose arbitrary additional work. The added work must belong to the problem and remain executable within memory limits.

## 3. Three models that must be distinguished

### A. Perfectly divisible work, zero overhead

If \(T_P=W/P\), then \(T_o=0\) and \(E=1\) for every feasible size and processor count. There is no nontrivial communication-driven isoefficiency requirement, aside from task granularity and availability. One cannot solve the exact-target equation for a lower \(E_0<1\) and obtain a positive required workload: this ideal model never has that lower efficiency.

### B. No added communication, but fixed useful serial work

Let \(T_P=A+(W-A)/P\), with constant \(A>0\). Although commonly called “overhead-free” in Amdahl/Gustafson derivations, it has

\[
T_o=(P-1)A.
\]

The idle capacity during the serial phase is overhead in the formal isoefficiency definition. At target \(E_0\),

\[
W=\frac{E_0}{1-E_0}(P-1)A=\Theta(P).
\]

This establishes the linear isoefficiency of **this particular fixed-\(A\) model**. It differs from requiring exact constant efficiency along the preselected path \(W=A+PB\); that path only approaches a constant. The \(P=1\) boundary has zero overhead and efficiency 1, so an exact lower target is meaningful here for sufficiently large \(P\), with \(W\geq A\).

### C. Exposed communication

If \(T_P=A+(W-A)/P+C(P)\), then

\[
T_o=(P-1)A+PC(P).
\]

For positive fixed coefficients and communication independent of \(W\):

| Exposed elapsed overhead | Required work at fixed \(E_0\), fixed \(A\) |
|---|---|
| \(C=c\log P\) | \(\Theta(P\log P)\) |
| \(C=d\sqrt P\) | \(\Theta(P^{3/2})\) |
| \(C=c\), constant | \(\Theta(P)\) |

These orders require \(\Theta\) behavior, not merely big-O upper bounds. They show why constant work per worker cannot maintain positive efficiency against divergent exposed communication. Maintaining \(E_0\) by growing work faster generally sacrifices constant walltime: \(T_P=W/(PE_0)\) grows like \(\log P\) or \(\sqrt P\) in the first two rows.

## 4. Why the workload dependence matters

Suppose serial setup grows as \(A(W)=aW^\gamma\), \(0<\gamma<1\), and other costs are negligible. Then \(T_o\sim aP W^\gamma\), so the target equation yields

\[
W^{1-\gamma}=\Theta(P),\qquad
W=\Theta(P^{1/(1-\gamma)}).
\]

Linear isoefficiency has disappeared. If a fixed positive fraction of useful work is intrinsically serial, increasing problem size cannot maintain a fixed positive efficiency for unbounded \(P\). Communication depending on data size creates further implicit constraints. Isoefficiency therefore requires a work/overhead model plus a feasible input-size mapping; Gustafson's ideal curve alone is insufficient.

## Sources

- [G88] Gustafson, original scaled-work argument: <https://doi.org/10.1145/42411.42415>.
- [KG91] V. Kumar and A. Gupta, “Analysis of scalability of parallel algorithms and architectures: a survey,” ICS 1991, 396–405: <https://doi.org/10.1145/109025.109118>. Bibliographic details verified through [Crossref](https://api.crossref.org/works/10.1145/109025.109118). Equations above are derived for the explicitly stated models.
