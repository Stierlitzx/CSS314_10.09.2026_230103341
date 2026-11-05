# Task 06 — Parametric Gustafson speedup and efficiency

**Student ID:** 230103341

## My model and calculation

I use P for processor count and α for the non-scaling fraction of the **P-processor elapsed runtime**, normalized to one. I assume equal-speed processors, balanced parallel work, no additional overhead, and the same scaled workload in the serial reference.

```text
Serial-reference time = α + P(1 − α)
S_G = α + P(1 − α) = P − α(P − 1)
E_G = S_G/P = 1 − α + α/P
f = α/S_G
```

Here S_G is scaled speedup, E_G is efficiency, and f is the corresponding serial-reference fraction—not α. I retain the assignment's α = 0.04 at every listed P; these are model predictions, not measurements.

```text
α = 0.04 = 1/25
S_G = (24P + 1)/25
E_G = (24P + 1)/(25P) = 0.96 + 0.04/P

At P = 1024:
S_G = 1024 − (1023/25) = 24577/25 = 983.08
E_G = 24577/25600 = 0.9600390625 = 96.00390625%
```

## My results

All speedup decimals and efficiency percentages below are exact.

| P | α | Exact S_G | S_G decimal | Exact E_G | Efficiency (%) |
|---:|---:|---|---:|---|---:|
| 16 | 0.04 | 77/5 | 15.40 | 77/80 | 96.25 |
| 64 | 0.04 | 1537/25 | 61.48 | 1537/1600 | 96.0625 |
| 256 | 0.04 | 1229/5 | 245.80 | 1229/1280 | 96.015625 |
| 1024 | 0.04 | 24577/25 | 983.08 | 24577/25600 | 96.00390625 |
| 4096 | 0.04 | 19661/5 | 3932.20 | 19661/20480 | 96.0009765625 |
| 16384 | 0.04 | 393217/25 | 15728.68 | 393217/409600 | 96.000244140625 |

I obtain linear speedup with slope 0.96 and intercept 0.04, while efficiency approaches 96% from above—not 100%.

```text
Ideal-speedup loss = P − S_G = α(P − 1)
At P = 16384: loss = 0.04 × 16383 = 655.32
Check: 16384 − 15728.68 = 655.32
As P → ∞: E_G → 1 − α = 0.96
```

I distinguish workload growth from dimension growth: if work scales as N^d, constant per-processor work requires dimension N to scale as P^(1/d), not P. I checked all six rows with exact rational arithmetic, including both speedup expressions, efficiency, and the bounds from 1 to P.