# Task 15 — Energy scaling requires matched useful work

**Student ID:** 230103341

## 1. Specify the meaning of active power

Let \(P\) denote the number of identical nodes, not power. Each powered node consumes baseline/idle power \(p_i\geq0\) watts and **incremental dynamic power** \(p_d\geq0\) watts when computing. Total active-node power is \(p_i+p_d\). If a specification instead calls the **total active power** \(p_a\), use \(p_d=p_a-p_i\), not \(p_i+p_a\). Adding idle power to an already-total active value double-counts it.

Energy is \(J=\int p(t)\,dt\). The constant-power models below are analytical illustrations; no measured watts or joules are claimed. Assume identical node speed, constant active power, and no frequency-dependent changes. Network, cooling and shared infrastructure are excluded initially and must be added consistently in a real comparison.

## 2. Equal walltime does not mean equal workload

Consider the ideal scaled family with serial useful phase \(A\), parallel useful phase \(B\) per worker and total parallel time \(T=A+B\). The expanded job \(W_P\) takes

\[
T_1(W_P)=A+PB=ST,\qquad
S=\alpha+P(1-\alpha),\quad \alpha=A/T.
\]

The original one-node job \(W_1\) takes \(T\). If every allocated node draws total active power throughout the run, then

\[
J_1(W_1)=(p_i+p_d)T,\qquad
J_P(W_P)=P(p_i+p_d)T.
\]

The ratio is \(P\), but the workloads differ. The parallel job performs \(S\) times as much reference useful work in the same time. Its useful work per joule relative to the small one-node job is \(S/P=E\), not \(S\). Equal-time weak scaling is a capacity/throughput comparison, not an energy-to-solution comparison for one fixed job.

## 3. Compare the same expanded job

Serial execution of \(W_P\) requires \(ST\), so under the same active power

\[
J_1(W_P)=(p_i+p_d)ST.
\]

If all \(P\) nodes remain fully active for the entire parallel run,

\[
\frac{J_P(W_P)}{J_1(W_P)}=\frac{P}{S}=\frac1E\geq1
\]

for ideal \(S\leq P\). Faster completion alone therefore does not prove an energy saving. This full-active assumption is an upper model for otherwise idle workers, not a claim that useful arithmetic occurs on every node during the serial phase.

A more phase-aware model keeps every node powered at baseline, but only one node draws incremental dynamic power during \(A\); all nodes compute during \(B\). Then

\[
J_P=Pp_iT+p_d(A+PB)=T(Pp_i+Sp_d),
\]

\[
\boxed{R_J=\frac{J_P}{J_1(W_P)}
=\frac{(P/S)p_i+p_d}{p_i+p_d}}.
\]

For \(p_i+p_d>0\), \(1\leq S\leq P\),

\[
\boxed{1\leq R_J\leq P/S}.
\]

Dynamic energy is identical to serial dynamic energy in this ideal work-conserving model; extra energy comes from baseline power on idle allocated nodes. If \(p_i=0\), the ratio is 1. If \(S=P\), it is also 1. Perfectly powering off unused nodes during the serial phase could remove their baseline contribution, but requires explicit assumptions about transition costs and availability.

## 4. Numerical example and a general saving criterion

Take illustrative values \(P=16\), \(\alpha=0.1\), \(T=10\) s, \(p_i=30\) W and \(p_d=70\) W. Then \(S=14.5\), \(E=0.90625\), and the matched serial job takes 145 s.

| Comparison quantity | Energy |
|---|---:|
| Small one-node job, 10 s | 1,000 J |
| Expanded job, one node, 145 s | 14,500 J |
| Expanded job, 16 fully active nodes, 10 s | 16,000 J |
| Expanded job, phase-aware power, 10 s | 14,950 J |

The matched-work ratios are 1.10345 and 1.03103, respectively. Neither equals the unmatched equal-time ratio 16.

More generally, let \(\bar p_P\) be average **total system** power for the parallel job and \(\bar p_1\) the corresponding serial power. Using the actual matched-work speedup \(S=T_1/T_P\),

\[
\frac{J_P}{J_1}=\frac{\bar p_P/\bar p_1}{S},\qquad
\boxed{J_P\leq J_1\iff\bar p_P/\bar p_1\leq S}.
\]

Thus energy saving is possible if power grows less than speedup—for example through lower voltage, sleeping idle nodes, or amortizing substantial shared baseline power. It is not guaranteed by Gustafson scaling. Communication, busy waiting and longer runtime can increase energy without adding useful work; real measurements should include these phases, clearly specify the power boundary, and report joules per matched solution or per validated useful result.

## Source and scope

The workload accounting follows Gustafson's original scaled-work interpretation: <https://doi.org/10.1145/42411.42415>. Power assumptions, energy equations and numerical examples here are derived analytical models, not hardware measurements or an empirical energy law.
