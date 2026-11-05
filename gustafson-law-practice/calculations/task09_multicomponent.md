# Task 09 — I/O, broadcast, synchronization, and useful parallel work

**Student ID:** 230103341

## Timing components and explicit modeling assumption

Normalize the elapsed time of the observed `P`-processor execution to one. Let the **non-overlapping critical-path elapsed-time fractions** be

* **Phase A:** initial disk I/O plus broadcast, `a=0.02=1/50`;
* **Phase B:** boundary synchronization, `y=0.06=3/50`;
* **Phase C:** useful parallel flux computation, `c=0.92=23/25`.

These are the handout's fractions of the **parallel runtime**. Assume they exhaust the timing interval and are counted once, so `a+y+c=0.02+0.06+0.92=1`. They must not be sums of CPU times across ranks or overlapping timers, which could double-count elapsed time.

For this simplified calculation, **all of I/O, broadcast, and synchronization are treated as non-scaling**, with combined fraction

\[
\alpha=a+y=0.02+0.06=0.08=\frac2{25},\qquad
c=1-\alpha=0.92=\frac{23}{25}.
\]

This is a stipulated effective timing model. These activities are **not inherently all serial**. Parallel I/O can distribute requests; broadcast is usually a collective with a tree or another concurrent algorithm; synchronization involves waiting and can depend on rank count and load imbalance. Their costs can decrease, increase, overlap, or change with `P`. Treating their total as a fixed `α=0.08` is an assumption, not a physical classification of every instruction.

Phase A's 2% is the combined disk-I/O-plus-broadcast time; the handout does not supply an internal split. Phase B contributes the additional 6%. No finer split is needed or invented.

## Two-component approximation

Assume equal processor rates, perfectly balanced scalable computation, and that the non-scaling component can be carried unchanged into the reference-time construction. Serial-equivalent reference time is

\[
T_{\rm ref}=a+y+Pc=\alpha+P(1-\alpha).
\]

With observed time normalized to one,

\[
\boxed{S_G=0.08+0.92P=\frac{2+23P}{25}},
\qquad
\boxed{E_G=0.92+\frac{0.08}{P}}.
\]

If broadcast or synchronization disappears on one processor, `T_ref` is a **model serial-equivalent reference**, not an exact prediction of an actual one-processor run. The performance metric must use a consistent timing scope. A true end-to-end speedup needs an actual serial-reference cost model or measurement.

## Substitution and numerical table

The assignment asks for `P=512` and `P=2048`, with the phase fractions assumed to apply at both counts. At `P=512`,

\[
S_G=0.02+0.06+512(0.92)
=0.08+471.04=\boxed{471.12=\frac{11778}{25}},
\]
\[
E_G=\frac{471.12}{512}=\frac{5889}{6400}
=0.92015625=92.015625\%.
\]

At `P=2048`,

\[
S_G=0.02+0.06+2048(0.92)
=0.08+1884.16=\boxed{1884.24=\frac{47106}{25}},
\]
\[
E_G=\frac{1884.24}{2048}=\frac{23553}{25600}
=0.9200390625=92.00390625\%.
\]

| P | Non-scaling contribution | Scalable serial-equivalent contribution 0.92P | Exact S_G | S_G decimal | Efficiency (%) |
|---:|---:|---:|---|---:|---:|
| 512 | 0.08 | 471.04 | 11778/25 | 471.12 | 92.015625 |
| 2048 | 0.08 | 1884.16 | 47106/25 | 1884.24 | 92.00390625 |

The losses from ideal speedup are `512−471.12=40.88=0.08×511` and `2048−1884.24=163.76=0.08×2047`. In this fixed-`α` model, asymptotic efficiency is 92%, not zero, while speedup grows without a finite constant ceiling.

## Why the serial-baseline fraction is different

The non-scaling **machine elapsed-time fraction** is `α=0.08`. Under the ideal matched-reference model, the corresponding fractions of the serial-reference time are

\[
f_{512}=\frac{2/25}{11778/25}=\frac1{5889},
\qquad
f_{2048}=\frac{2/25}{47106/25}=\frac1{23553}.
\]

Neither is 8%. Substituting `f=0.08` directly into Amdahl's fixed-work formula would describe a **different reference scenario**. With the converted matched fractions,

\[
S_A(512;1/5889)=\frac{11778}{25}=471.12,
\quad
S_A(2048;1/23553)=\frac{47106}{25}=1884.24.
\]

## More general component model

For non-overlapping phases with actual cost functions,

\[
T_P(N)=T_{\rm IO}(P,N)+T_{\rm bcast}(P,N)
+T_{\rm sync}(P,N)+\frac{W(N)}{Pr},
\]

where `r` is each processor's useful work rate. Then a model-defined effective fraction is

\[
\alpha(P,N)=\frac{T_{\rm IO}+T_{\rm bcast}+T_{\rm sync}}{T_P(N)}.
\]

Changing `P` or problem dimension `N` changes both numerator and denominator. Overlap requires a critical-path model rather than this simple sum. Consequently `α=0.08` should not be extrapolated without checking timing data and implementation behavior.

## Arithmetic validation log

Python `fractions.Fraction` verified all exact speedups, efficiencies, the loss identity, and the conversion of the reference fraction.

```text
Phase A=1/50, Phase B=3/50, Phase C=23/25; sum=1
alpha=A+B=2/25; scalable fraction=23/25
P=512:  S_G=11778/25=471.12; E_G=5889/6400=.92015625
P=2048: S_G=47106/25=1884.24;E_G=23553/25600=.9200390625
P=512:  alpha/S_G=1/5889;  matched Amdahl speedup=11778/25
P=2048: alpha/S_G=1/23553; matched Amdahl speedup=47106/25. PASS
```

**Conclusion:** `S_G=0.08+0.92P` follows only after explicitly grouping I/O, broadcast, and synchronization into the assumed non-scaling 8%. It does not establish that these operations are intrinsically serial.