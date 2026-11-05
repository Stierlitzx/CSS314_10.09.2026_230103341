# Task 07 — Maximum tolerable non-scaling fraction

**Student ID:** 230103341

## My inverse calculation

I use P for processor count, α for the non-scaling fraction of the **P-processor elapsed runtime**, S_G for Gustafson speedup, and E_G for efficiency. I assume balanced parallel work, equal processor rates, and no unmodeled overhead.

For a target speedup S_* and P > 1, I rearrange:

```text
S_G = P − α(P − 1); E_G = S_G/P
S_G ≥ S_*  ⇒  α(P − 1) ≤ P − S_*
α ≤ α_max = (P − S_*)/(P − 1)

For target efficiency η, S_* = ηP:
α_max = P(1 − η)/(P − 1)
```

The speedup bound applies between 1 and P. Targets above P are impossible; targets at most 1 admit every α between zero and one. At P = 1, speedup is always 1 and the inverse denominator vanishes. Similarly, efficiency above 1 is impossible, while efficiency at most 1/P admits every α. An ideal efficiency target requires α = 0; as P grows, α_max approaches 1 − η.

## My answer for 8192 processors

The assignment requires speedup of at least 7500:

```text
8192 − 8191α ≥ 7500
0 ≤ α ≤ 692/8191
α_max = 692/8191 = 0.08448296911244048…
      ≈ 8.448296911244048%

At equality:
S_G = 8192 − 692 = 7500
E_G = 7500/8192 = 1875/2048
    = 0.91552734375 = 91.552734375%

For any admissible ε > 0:
α = 692/8191 + ε  ⇒  S_G = 7500 − 8191ε < 7500
```

I therefore report the exact fraction as the maximum; rounding upward could miss the target.

## My reference-fraction check

This threshold describes the parallel run, not the serial-reference fraction f. For the matched two-component workload:

```text
f = α_max/7500 = 173/15358125
  = 0.00001126439588165873…
  ≈ 0.001126439588165873%
Amdahl inverse: f_max = (8192/7500 − 1)/8191
                     = 173/15358125
```

I checked equality and the fraction conversion with exact rational arithmetic. Both laws give 7500 after using their respective reference fractions.