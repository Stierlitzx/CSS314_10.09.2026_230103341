# Task 07 — Maximum tolerable non-scaling fraction

**Student ID:** 230103341

## Definitions and model

Let `P>1` be the processor count and `α∈[0,1]` the non-scaling fraction of the **P-processor elapsed runtime**, not the serial fraction `f` of the one-processor reference. Assume balanced useful parallel work, equal processor rates, and no unmodeled overhead. Gustafson speedup and efficiency are

\[
S_G=P-\alpha(P-1),\qquad E_G=\frac{S_G}P.
\]

## General inverse for a target speedup

For target `S_*` and `1≤S_*≤P`,

\[
P-\alpha(P-1)\ge S_*
\Longleftrightarrow
\alpha(P-1)\le P-S_*
\Longleftrightarrow
\boxed{\alpha\le\alpha_{\max}=\frac{P-S_*}{P-1}}.
\]

The inequality direction is unchanged on division because `P−1>0`. A negative bound means the target exceeds ideal speedup and is impossible in this model. If `S_*≤1`, every `α∈[0,1]` satisfies it. At `P=1`, speedup is always 1 and the inverse formula must not be used because its denominator vanishes.

For a target efficiency `η`, substitute `S_*=ηP`:

\[
\boxed{\alpha_{\max}=\frac{P(1-\eta)}{P-1}}.
\]

For `1/P≤η≤1` this lies in `[0,1]`. If `η>1` the target is impossible; if `η≤1/P` all admissible fractions pass. As `P→∞`, `α_max→1−η`. For an exactly ideal target `η=1`, only `α=0` passes.

## Assignment substitution: P = 8192 and S_G ≥ 7500

The handout requires speedup at least 7500 on 8192 processors. Substitute directly:

\[
8192-\alpha(8192-1)\ge7500
\Longleftrightarrow8191\alpha\le692
\Longleftrightarrow
\boxed{0\le\alpha\le\frac{692}{8191}}.
\]

Therefore the maximum tolerable non-scaling fraction is

\[
\boxed{\alpha_{\max}=\frac{692}{8191}
=0.08448296911244048\ldots
\approx8.448296911244048\%}.
\]

The exact rational bound is authoritative: rounding it upward would permit fractions that miss the target. At equality,

\[
S_G=8192-\frac{692}{8191}(8191)=8192-692=7500,
\]
\[
E_G=\frac{7500}{8192}=\frac{1875}{2048}
=0.91552734375=91.552734375\%.
\]

For any admissible positive increment `ε`, choosing `α=692/8191+ε` gives `S_G=7500−8191ε<7500`. This proves that the threshold is maximal, not merely sufficient.

## Reference-fraction caveat

The 8.448296911244048% threshold is a fraction of the **8192-processor run**, not the serial fraction of a fixed one-processor baseline. At equality, the matched serial-reference fraction is

\[
f=\frac{\alpha}{S_G}=\frac{692/8191}{7500}
=\frac{173}{15358125}
=0.00001126439588165873\ldots,
\]

or approximately 0.001126439588165873%. Amdahl's inverse for that matched target independently gives

\[
f_{\max}=\frac{P/S_*-1}{P-1}
=\frac{8192/7500-1}{8191}
=\frac{173}{15358125},
\]

which agrees after converting the reference fractions. This is a consistency check, not a substitution of baseline `f` for machine fraction `α`.

## Arithmetic validation log

Python exact rational arithmetic checked substitution at equality and verified that increasing `α` by a positive amount lowers speedup below the corresponding target.

```text
P=8192, target=7500
alpha_max=(8192-7500)/(8192-1)=692/8191=.08448296911244048...
S_G(alpha_max)=7500 exactly; E_G=1875/2048=.91552734375
alpha_max + epsilon => S_G = 7500 - 8191*epsilon < 7500 for epsilon>0
f=alpha_max/7500=173/15358125; Amdahl with converted f gives 7500
All equality and fraction-conversion checks PASS.
```