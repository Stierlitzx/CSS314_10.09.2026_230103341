# Task 12 — Communication as an elapsed-time penalty

**Student ID:** 230103341

## 1. Count useful work independently of overhead

Consider a scaled workload with useful serial time \(A\geq0\) and useful parallel work costing \(B>0\) seconds **per worker**, constant as \(P\) grows. The ideal parallel time is \(T_0=A+B\); serial execution of this same scaled workload takes \(A+PB\). Let \(C(P)\geq0\) be additional exposed communication/synchronization time on the critical path, measured in seconds. Then

\[
T_P=A+B+C(P),\qquad
\boxed{S_{\mathrm{useful}}(P)=\frac{A+PB}{A+B+C(P)}}.
\]

The reference serial algorithm is assumed not to need this distributed communication. Consequently overhead belongs in the denominator, **not in the useful-work numerator**. If communication overlaps computation, \(C\) means only its unhidden contribution; adding full transfer time would double-count overlap. Setup or imbalance can be modeled separately or included in a clearly labeled total overhead.

Writing \(\alpha=A/T_0\), \(c(P)=C(P)/T_0\),

\[
S_{\mathrm{useful}}
=\frac{\alpha+P(1-\alpha)}{1+c(P)},\qquad
E=\frac{S_{\mathrm{useful}}}{P}
=\frac{1-\alpha+\alpha/P}{1+c(P)}.
\]

The fraction \(\alpha\) is relative to useful ideal elapsed time. It is not \(A/T_P\) once communication is added. Applying the ordinary Gustafson formula to \(A/T_P\) and treating all remaining elapsed time as useful parallel work would falsely credit communication as computation. The ideal expression itself comes from reconstructing serial useful work, not from subtracting an arbitrary “communication speedup penalty.” [G88]

## 2. Two explicit growth models

A statement \(C(P)=O(\log P)\) is only an upper bound: zero or bounded cost also satisfies it. To obtain the following limits, assume positive coefficients and the stated **equalities**, or corresponding \(\Theta\) growth.

### Tree-like latency

Let \(C(P)=c\log_2P\), \(c>0\), for powers of two. Then

\[
S=\frac{A+PB}{T_0+c\log_2P}
=\Theta(P/\log P),\quad
E=\Theta(1/\log P)\longrightarrow0.
\]

Speedup grows without bound but efficiency falls: more total work is done, with less useful throughput per worker. A fixed fan-in tree with a fixed-size message motivates this model; it is not a universal cost law for every collective.

### Square-root communication

Let \(C(P)=d(\sqrt P-1)\), \(d>0\). The subtraction only enforces zero extra cost at \(P=1\); communication is still added to runtime. Now

\[
S=\Theta(\sqrt P),\qquad E=\Theta(P^{-1/2})\longrightarrow0.
\]

An exposed cost with this growth might model a topology-dependent exchange; it must be justified for the particular algorithm and network rather than inferred from processor count alone.

More generally, at constant \(A,B\), any \(C(P)\to\infty\) forces \(E\to0\). Bounded overhead \(C\to C_\infty\) instead gives \(E\to B/(A+B+C_\infty)\). If \(C=\Theta(P)\), speedup remains bounded; superlinear \(C\) can eventually make speedup tend to zero.

## 3. Illustrative calculation and overhead budget

Take \(A=1\) s, \(B=9\) s and \(P=64\). Useful serial time is 577 s and ideal parallel time is 10 s. The following numbers are analytical examples, not network measurements.

| Exposed cost model | \(C(64)\), s | \(T_{64}\), s | Useful speedup | Efficiency |
|---|---:|---:|---:|---:|
| No overhead | 0 | 10 | 57.700 | 0.90156 |
| \(0.1\log_2P\) s | 0.6 | 10.6 | 54.434 | 0.85053 |
| \(0.1(\sqrt P-1)\) s | 0.7 | 10.7 | 53.925 | 0.84258 |

For target efficiency \(E_0>0\), rearrangement gives the allowable exposed communication budget:

\[
C(P)\leq\frac{B+A/P}{E_0}-(A+B).
\]

A negative right-hand side means the target is impossible even without communication at that processor count. With fixed useful per-worker work, the budget approaches a constant; divergent communication eventually violates any fixed positive target.

## 4. What a real experiment must report

Measure useful computation, collective time, imbalance, message volume and total elapsed time without summing overlapping intervals. Preserve the same useful result in the serial reference. If per-worker work is increased to hide growing communication, label this as a new workload family: it may restore efficiency but is no longer the constant-\(B\) family above. Report both useful speedup and elapsed time, since the ideal fixed-time premise has been lost whenever \(C(P)\) grows.

## Source

- [G88] J. L. Gustafson, “Reevaluating Amdahl's Law” (1988), original scaled-work interpretation: <https://doi.org/10.1145/42411.42415>. Communication equations and examples here are explicit analytical extensions, not claims from that paper.
