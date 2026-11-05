# Task 11 — Gustafson versus Sun–Ni memory-bounded scaling

**Student ID:** 230103341

## 1. Define the baseline before comparing curves

Sun and Ni's memory-bounded approach lets the workload expand with available memory, rather than assuming either fixed work or fixed execution time. Their original paper is *Scalable Problems and Memory-Bounded Speedup*, Journal of Parallel and Distributed Computing 19(1), 27–37 (1993). [SN93] The following is a normalized analytical restatement, not a measured scaling result.

Normalize the one-processor **baseline workload** to serial time 1. Let its serial fraction be \(s\), so its serial and parallelizable work cost \(s\) and \(1-s\). On \(P\) identical processors, retain serial work \(s\) and expand parallelizable work by a factor \(G(P)\). With perfect balance and no additional overhead,

\[
T_1(W_P)=s+(1-s)G(P),\qquad
T_P(W_P)=s+\frac{(1-s)G(P)}{P},
\]

\[
\boxed{S_{SN}(P)=
\frac{s+(1-s)G(P)}{s+(1-s)G(P)/P}}.
\]

Here \(s\) is the baseline single-processor fraction, **not** the serial fraction observed during the expanded parallel run. The numerator is the serial time for the expanded workload \(W_P\), not for the original input. Assume \(G(1)=1\) and \(0\leq s\leq1\); exclude the degenerate zero-work case.

Memory expansion does not by itself specify work expansion. If parallel work is proportional to \(N^q\), storage to \(N^r\), and usable aggregate memory grows by \(P\), then filling that memory gives \(N_P/N_1=P^{1/r}\) and \(G(P)=P^{q/r}\). Replicated data, temporary storage and communication buffers can invalidate this simple mapping.

## 2. Recover the familiar laws

- **Fixed work:** \(G=1\) gives Amdahl's law,
  \[
  S=\frac{1}{s+(1-s)/P}.
  \]
- **Linear parallel-work expansion:** \(G=P\) gives
  \[
  T_P=s+(1-s)=1,\qquad S=s+P(1-s).
  \]
  This is the Gustafson expression, now derived from a specified baseline and growth path.
- **General growth:** the parallel-observed serial fraction is
  \[
  \alpha_P=\frac{s}{s+(1-s)G/P},
  \qquad S=\alpha_P+P(1-\alpha_P).
  \]
  This last identity holds for every ideal workload in this model, but \(\alpha_P\) need not be constant across processor counts.

Gustafson's fixed-time interpretation defines the fraction from the parallel execution. [G88] If the same symbol and numerical value are used for \(s\) and a fixed \(\alpha\), the \(G=P\) numerical curves coincide. That coincidence does not make their workload baselines interchangeable: \(s\) describes the original serial execution, whereas \(\alpha\) describes the scaled parallel execution. For general \(G\), equating them is incorrect.

## 3. When does the linear-growth prediction over- or underestimate?

For \(P>1\) and \(0<s<1\), differentiation gives

\[
\frac{\partial S_{SN}}{\partial G}
=\frac{s(1-s)(1-1/P)}{[s+(1-s)G/P]^2}>0.
\]

Thus, **holding the baseline \(s\) fixed**, substituting \(G=P\) overestimates ideal scaled speedup if the feasible expansion is \(G<P\), and underestimates it if \(G>P\). This is a comparison of different workload families, not evidence that one machine executes a fixed job faster merely because it has more memory.

For example, at \(s=0.1\), \(P=16\):

| Parallel-work expansion | Ideal parallel time | Scaled speedup |
|---|---:|---:|
| \(G=1\) | 0.15625 | 6.4000 |
| \(G=4\) | 0.32500 | 11.3846 |
| \(G=16\) | 1.00000 | 14.5000 |
| \(G=64\) | 3.70000 | 15.5946 |

These are calculated examples, not benchmarks. The last row buys a higher scaled speedup with a longer runtime; it does not satisfy the same fixed-time budget. For any positive workload the ideal model still has \(S\leq P\). When \(s=0\), all positive \(G\) give \(S=P\); when \(s=1\), expansion adds no work and \(S=1\).

## 4. Practical interpretation

Sun–Ni is useful when memory capacity determines feasible scientific resolution. Gustafson is useful when a time budget determines how much useful work to attempt. Neither makes communication disappear or proves that serial setup remains invariant. A real comparison must state \(M(N)\), useful work \(W(N)\), serial work \(A(N)\), and communication cost separately. Growing serial initialization or replicated memory can make the ideal memory-bounded prediction optimistic even when aggregate physical memory grows linearly.

## Sources

- [SN93] X.-H. Sun and L. M. Ni, original memory-bounded model: <https://doi.org/10.1006/jpdc.1993.1087>. Bibliographic metadata verified through [Crossref](https://api.crossref.org/works/10.1006/jpdc.1993.1087); publisher full text was not accessible during preparation.
- [G88] J. L. Gustafson, “Reevaluating Amdahl's Law,” CACM 31(5), 532–533 (1988): <https://doi.org/10.1145/42411.42415>.
