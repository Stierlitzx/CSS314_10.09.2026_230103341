# Task 17 — Three-dimensional stencil scaling and halos

**Student ID:** 230103341

## My work and geometry model

I use global cubic-grid edge N, P ranks (one per processing element), fixed halo width h, and k operations per owned cell per step. I assume balanced ownership, nearest-neighbor communication, enough memory, and fixed step count. Stability or convergence requiring more steps would change work growth.

```text
W(N) = kN³ per step; work per rank = kN³/P
Constant per-rank work: kN³/P = kN₁³ ⇒ N = N₁P^(1/3)
```

Here N₁ is the one-rank edge. Cells, work, and owned-cell memory grow as P. I distinguish the global exterior from internal interfaces:

```text
Global exterior area = 6N²; exterior area/volume = 6/N
For P = q³ cubic ranks: local edge ℓ = N/q = N/P^(1/3)
Six-face halo per rank ≈ 6hℓ²
H_sum ≈ 6hP(N/P^(1/3))² = 6hN²P^(1/3)
Nonperiodic unique internal interface area = 3(q − 1)N²
Nonperiodic directed halo transfers = 6h(q − 1)N²
```

H_sum counts local faces, not merely the exterior. Directed transfers count both sides of each internal interface. The six-face estimate omits edge/corner corrections and applies to periodic ranks; one-rank copies may be local. Nonperiodic boundaries reduce neighbors but not the asymptotic scaling.

## My cubic scaling results

```text
Cell-count ratio R_cube = 6hℓ²/ℓ³ = 6hP^(1/3)/N
Strong scaling (fixed N): R_cube grows as P^(1/3)
Weak scaling: ℓ = N₁, R_cube = 6h/N₁, constant
```

I obtain constant local halo/computation ratio, not improvement, despite the falling global exterior ratio. Total halos and arithmetic both grow as P.

| P | N/N₁ | Cells / N₁³ | Exterior / (6N₁²) | ℓ/N₁ | Sum of rank face areas / (6N₁²) | Local halo/owned ratio |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1 | 1 | 1 | 6h/N₁ |
| 8 | 2 | 8 | 4 | 1 | 8 | 6h/N₁ |
| 64 | 4 | 64 | 16 | 1 | 64 | 6h/N₁ |
| 512 | 8 | 512 | 64 | 1 | 512 | 6h/N₁ |

For illustrative N₁ = 100 and h = 1, I calculate 1,000,000 owned cells and 60,000 face-halo cells per rank: 6%. Full padding instead adds 61,208 cells, or 6.1208%, including edges/corners.

For compute cost c seconds per cell, latency λ per message, and transfer cost β seconds per halo cell:

```text
T_comp = cℓ³
T_comm ≈ 6λ + 6βhℓ²
T_comm/T_comp ≈ (6λ + 6βhℓ²)/(cℓ³)
```

This ratio is ideally constant at fixed ℓ; contention, topology, collectives, and overlap can change actual time.

## My slab comparison

For local slabs with two communicating faces:

```text
Local shape = N × N × (N/P)
R_slab = 2hN²/(N³/P) = 2hP/N = (2h/N₁)P^(2/3)
Thickness = N/P = N₁P^(−2/3)
```

| P | Global N | Ideal slab thickness | Slab halo/owned ratio (2P/N) | Cubic six-face ratio |
|---:|---:|---:|---:|---:|
| 1 | 100 | 100 | 0.02 | 0.06 |
| 8 | 200 | 25 | 0.08 | 0.06 |
| 64 | 400 | 6.25 | 0.32 | 0.06 |
| 512 | 800 | 1.5625 | 1.28 | 0.06 |

I treat these as continuous geometry estimates, not implementable fractional layers or one-rank network traffic. Integer layers need balancing; slabs thinner than the halo/stencil radius invalidate the model. One nonperiodic rank has no inter-rank traffic.

Finally, for non-scaling fraction α of parallel elapsed time:

```text
S_G = P − α(P − 1)
```

I cannot infer constant α from constant local computation: concurrent exchanges and P-dependent synchronization need explicit timing scope. I checked both tables and the padded count exactly; these are analytical counts, not benchmarks.