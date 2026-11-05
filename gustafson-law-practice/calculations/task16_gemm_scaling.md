# Task 16 — GEMM dimension, work, and storage scaling

**Student ID:** 230103341

## My constant-compute-time model

I consider classical square multiplication C = AB, with global order N, one-processor order N₁, and P equal-speed processors. I assume balanced arithmetic, unchanged processing rate, and negligible communication, synchronization, startup, and serial overhead. Integer dimensions and block sizes may require rounding; Strassen-like algorithms are excluded.

```text
Leading work: W(N) ≈ 2N³ FLOPs
Constant work per processor: 2N³/P = 2N₁³
Therefore N = N₁P^(1/3)
```

I count multiplication and addition as two FLOPs. A literal dot-product implementation instead performs exactly 2N³ − N² operations. My cube-root rule holds per-processor work constant at leading order: dimension grows as P^(1/3), entries as P^(2/3), and work as P.

If I used square-root dimension growth, I would obtain memory weak scaling, not constant compute time:

```text
N = N₁sqrt(P) ⇒ N²/P = N₁²
W/W₁ ≈ P^(3/2); W/(PW₁) ≈ sqrt(P)
```

## My storage and result tables

With three resident matrices, b bytes per element, and no replication, workspace, or buffers, I obtain:

```text
W₁ = 2N₁³; M₁ = 3bN₁²; T₁ = baseline ideal compute time
M = 3bN² = M₁P^(2/3); M/P = M₁P^(−1/3)
At P = 64: N = 4N₁, W ≈ 128N₁³ = 64W₁, M = 16M₁
```

| P | N/N₁ | Leading W/W₁ | Work per processor / W₁ | M/M₁ | Memory per processor / M₁ | Ideal time / T₁ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | 2 | 8 | 1 | 4 | 1/2 | 1 |
| 64 | 4 | 64 | 1 | 16 | 1/4 | 1 |
| 512 | 8 | 512 | 1 | 64 | 1/8 | 1 |

I use N₁ = 1000 and b = 8 only as an illustration, not supplied measurements.

| P | Global N | Leading 2N³ (FLOPs) | Three-matrix bytes | Average bytes per processor |
|---:|---:|---:|---:|---:|
| 1 | 1,000 | 2,000,000,000 | 24,000,000 | 24,000,000 |
| 8 | 2,000 | 16,000,000,000 | 96,000,000 | 12,000,000 |
| 64 | 4,000 | 128,000,000,000 | 384,000,000 | 6,000,000 |
| 512 | 8,000 | 1,024,000,000,000 | 1,536,000,000 | 3,000,000 |

For strict equality under literal counting, I would solve the lower-order equation and round appropriately:

```text
At N = N₁P^(1/3): W_exact/P = 2N₁³ − N₁²P^(−1/3)
Strict target: (2N³ − N²)/P = 2N₁³ − N₁²
```

## My interpretation

For parallel elapsed-time non-scaling fraction α, speedup S_G, efficiency E_G, and matched serial fraction f:

```text
S_G = P − α(P − 1); E_G = S_G/P
f = α/[α + P(1 − α)]
```

I do not infer fixed α or constant wall time from kernel scaling. Communication, replicated panels, redistribution, and buffers invalidate ideal storage averages; shrinking local storage can reduce arithmetic intensity. I checked both tables exactly: every illustrative row has 2,000,000,000 leading FLOPs per processor.