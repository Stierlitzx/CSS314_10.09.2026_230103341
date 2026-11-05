# Task 17 — Three-dimensional stencil scaling and halo geometry

**Student ID:** 230103341

## Model and assumptions

Let `N` be the global linear dimension of a cubic `N × N × N` grid, `P` the number of ranks (one rank per processing element for this model), `h` a fixed halo width, and `k` the constant arithmetic work per owned cell per time step. Assume a fixed number of steps, nearest-neighbor communication, balanced ownership, and enough memory. If the number of steps grows with `N` for stability or convergence, the work model must also include that growth.

For one step,

\[
W(N)=kN^3,\quad W_{\rm rank}=kN^3/P.
\]

Ideal constant-work-per-rank weak scaling therefore requires

\[
\frac{kN^3}P=kN_1^3
\Longrightarrow\boxed{N=N_1P^{1/3}}.
\]

The linear dimension grows as a cube root, while global cell count, arithmetic work, and owned-cell memory grow linearly in `P`.

## Global outer surface is not total inter-rank halo traffic

The global cube's exterior has area proportional to `N²` (six faces give `6N²`). Its area/volume ratio is `6/N`. Under weak scaling that ratio decreases as `P^(−1/3)`, but this counts only the **outer boundary**, not the internal partition interfaces.

For `P=q³` ranks arranged as `q × q × q` cubes, local edge length is

\[
\ell=N/q=N/P^{1/3}.
\]

Ignoring edge/corner corrections, the six face halos per rank contain about `6hℓ²` cells. Summing over ranks counts

\[
H_{\rm sum}\simeq6hP\left(\frac{N}{P^{1/3}}\right)^2
=6hN^2P^{1/3}.
\]

The essential geometry is **`P × (N/P^(1/3))²`**, not merely `N²`. Counting both send directions gives two copies of each internal interface, which is appropriate for directed halo exchanges. Constants depend on boundary conditions and on whether send, receive, or stored halo cells are counted.

For periodic boundaries and face-only transfers, the six-face model applies to all ranks (with the special one-rank case possibly handled as local copies). For nonperiodic boundaries, there are `3(q−1)N²` unique internal interface cells across the three orientations and

\[
H_{\rm directed}=6h(q-1)N^2
\]

directed cell transfers per step, excluding edge/corner transfers. The large-`q` scaling agrees with `6hN²P^(1/3)`. Fewer neighbors on physical boundaries cause finite-size corrections, not an improving asymptotic per-rank surface/volume ratio.

## Cubic decomposition: strong versus weak scaling

The leading per-rank communication/computation **cell-count** ratio is

\[
R_{\rm cube}=\frac{6h\ell^2}{\ell^3}
=\frac{6h}{\ell}=\frac{6hP^{1/3}}N.
\]

* **Strong scaling (fixed N):** `R_cube ∝ P^(1/3)` worsens as local cubes shrink.
* **Ideal cubic weak scaling (`N=N₁P^(1/3)`):** `ℓ=N₁`, so `R_cube=6h/N₁` is **constant, not improving**. Total halo data and total arithmetic both grow as `P`.

A time ratio additionally needs costs: if compute time is `cℓ³` and communication time is approximately `6λ + 6βhℓ²`, where `λ` is message latency and `β` seconds per transferred halo cell, then

\[
\frac{T_{\rm comm}}{T_{\rm comp}}
\simeq\frac{6\lambda+6\beta h\ell^2}{c\ell^3}.
\]

With constant local cube size this idealized ratio is constant as well. Network contention, topology, collectives, and overlap can change observed time, so geometry alone does not prove perfect weak scaling.

### Exact multiplier table

The following table uses leading geometry, omits the factor `6h`, and sets `N=N₁P^(1/3)`. `H_sum` counts the sum of local face areas, not just the global exterior.

| P | N/N₁ | Global cells / N₁³ | Global exterior area / (6N₁²) | Local edge ℓ/N₁ | Sum of rank face areas / (6N₁²) | Local halo/owned-cell ratio |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1 | 1 | 1 | `6h/N₁` |
| 8 | 2 | 8 | 4 | 1 | 8 | `6h/N₁` |
| 64 | 4 | 64 | 16 | 1 | 64 | `6h/N₁` |
| 512 | 8 | 512 | 64 | 1 | 512 | `6h/N₁` |

For an **illustrative** `N₁=100`, `h=1`, cubic weak scaling gives local owned volume `100³ = 1,000,000` cells and face-halo count `6×100² = 60,000` cells per rank: `60,000/1,000,000 = 0.06 = 6%` at every `P` in this ideal six-face model. A fully padded array instead has `(100+2)³−100³ = 61,208` halo cells, giving `6.1208%`; this includes edges and corners. Neither percentage improves just because more equal-sized cubes are added.

## Slab decomposition worsens under the same weak scaling

Split along one axis into local shapes `N × N × (N/P)`. Each interior/periodic slab exchanges two large faces, approximately `2hN²` cells, while owning `N³/P` cells. Therefore

\[
R_{\rm slab}=\frac{2hN^2}{N^3/P}
=\frac{2hP}N
=\frac{2h}{N_1}P^{2/3}
\quad\text{when }N=N_1P^{1/3}.
\]

Its local volume remains constant, but its shape becomes wide and thin: `N/P = N₁P^(−2/3)`. Halo traffic per rank grows as `P^(2/3)`. With the same illustrative `N₁=100`, `h=1`:

| P | Global N | Ideal slab thickness N/P | Slab halo/owned ratio `2P/N` | Cubic six-face halo/owned ratio |
|---:|---:|---:|---:|---:|
| 1 | 100 | 100 | 0.02 | 0.06 |
| 8 | 200 | 25 | 0.08 | 0.06 |
| 64 | 400 | 6.25 | 0.32 | 0.06 |
| 512 | 800 | 1.5625 | 1.28 | 0.06 |

These slab entries are **continuous geometry estimates** for two communicating faces, not claims that fractional cell thicknesses are implementable or that one rank needs network messages. Integer layers require rounding and load-balance adjustments. Once slabs become too thin for the halo width or the stencil radius, the simple model becomes impractical or invalid. A one-rank nonperiodic domain has no inter-rank traffic.

## Gustafson interpretation

For a non-scaling elapsed-time fraction `α` measured on the `P`-rank run, the ideal two-component scaled speedup is `S_G=P−α(P−1)`. Constant local computation does **not** ensure that communication is zero, serial, or a constant global fraction. Nearest-neighbor exchange can execute concurrently, while collectives and synchronization may introduce `P`-dependent critical-path costs. Any reduction to one `α` must state what timing components it includes.

## Arithmetic validation log

Python exact arithmetic checked the cube multipliers, halo counts, and slab ratios below. These are analytical counts, not measured performance.

```text
P:                         1       8       64       512
N/N1:                      1       2        4         8
outer area multiplier:     1       4       16        64
sum-rank-area multiplier:  1       8       64       512
cubic ratio (N1=100,h=1): 0.06    0.06     0.06      0.06
slab ratio (same inputs): 0.02    0.08     0.32      1.28
102^3 - 100^3 = 61208; 61208/1000000 = 0.061208. PASS
```

**Conclusion:** a falling global exterior surface/volume ratio is not evidence that distributed halo overhead improves. Cubic weak scaling holds the ideal local halo/computation ratio constant; slabs worsen it under the same global cube-root growth rule.