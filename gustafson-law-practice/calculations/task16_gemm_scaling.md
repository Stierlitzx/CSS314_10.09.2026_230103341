# Task 16 — Scaling the dimension, work, and storage of GEMM

**Student ID:** 230103341

## Model and assumptions

Consider classical dense square matrix multiplication `C = AB`, with all matrices of order `N`. Let `N₁` be the order handled by one processing element and `P` the number of equal-speed processing elements. Assume balanced division of classical arithmetic, unchanged arithmetic rate, and negligible communication, synchronization, startup, and serial overhead for the **ideal constant-compute-time derivation**. An integer dimension may require rounding and a particular implementation may impose block-size constraints.

Classical GEMM has leading-order work

\[
W(N)\simeq 2N^3\ \text{FLOPs}.
\]

Here a multiply and an addition count as two FLOPs. A literal dot-product implementation uses `N³` multiplies and `N²(N−1)` adds, hence exactly `2N³−N²` operations; `2N³` is the standard leading-order convention, not an exact count for every implementation. Strassen-like algorithms are outside this model.

## Constant per-processor work

Keeping arithmetic time approximately fixed requires `W(N)/P = W(N₁)` in the leading-order model:

\[
\frac{2N^3}{P}=2N_1^3
\Longrightarrow N^3=PN_1^3
\Longrightarrow\boxed{N=N_1P^{1/3}}.
\]

Thus the **problem dimension** grows as `P^(1/3)`, the **number of matrix entries** grows as `P^(2/3)`, and the **arithmetic work** grows as `P`. These are different quantities.

Using `N = N₁ sqrt(P)` would instead keep matrix storage per processor constant under ideal distribution:

\[
\frac{N^2}{P}=N_1^2,
\quad \frac{W(N)}{W(N_1)}=P^{3/2},
\quad \frac{W(N)}{P W(N_1)}=P^{1/2}.
\]

It would increase per-processor compute work and ideal compute time by `sqrt(P)`. It is **memory weak scaling, not constant-compute-time weak scaling for classical GEMM**.

## Storage and worked table

If three matrices are resident and each element occupies `b` bytes, without replication, workspace, or communication buffers,

\[
M(N)=3bN^2=3bN_1^2P^{2/3},\qquad
\frac{M(N)}P=3bN_1^2P^{-1/3}.
\]

At `P=64`, for example,

\[
N=N_1\sqrt[3]{64}=4N_1,\quad
W\simeq 2(4N_1)^3=128N_1^3=64W_1,
\quad M=3b(4N_1)^2=16M_1.
\]

The table uses perfect cubes to make every dimension multiplier exact. It is a parameter table in `N₁`, not a claim that a numerical baseline order was supplied.

| P | N/N₁ | Work W/W₁, leading order | Work per processor / W₁ | Total memory M/M₁ | Memory per processor / M₁ | Ideal compute time / T₁ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | 2 | 8 | 1 | 4 | 1/2 | 1 |
| 64 | 4 | 64 | 1 | 16 | 1/4 | 1 |
| 512 | 8 | 512 | 1 | 64 | 1/8 | 1 |

For an **illustrative**, not inferred, baseline `N₁ = 1,000` and `b = 8` bytes, the same substitutions give:

| P | Global order N | Leading work 2N³ (FLOPs) | Three-matrix storage (bytes) | Ideal average storage per processor (bytes) |
|---:|---:|---:|---:|---:|
| 1 | 1,000 | 2,000,000,000 | 24,000,000 | 24,000,000 |
| 8 | 2,000 | 16,000,000,000 | 96,000,000 | 12,000,000 |
| 64 | 4,000 | 128,000,000,000 | 384,000,000 | 6,000,000 |
| 512 | 8,000 | 1,024,000,000,000 | 1,536,000,000 | 3,000,000 |

For the literal exact operation count, `N=N₁P^(1/3)` gives

\[
\frac{W_{\rm exact}(N)}P=2N_1^3-N_1^2P^{-1/3},
\]

which is only asymptotically equal to the one-processor count `2N₁³−N₁²`. If strict equality for that exact counting convention is required, solve `(2N³−N²)/P = 2N₁³−N₁²` and handle integer rounding. This lower-order distinction does not change the cube-root scaling exponent.

## Gustafson interpretation and practical limits

If the measured non-scaling fraction **of the P-processor elapsed time** is `α`, Gustafson's ideal scaled speedup is

\[
S_G(P)=\alpha+P(1-\alpha)=P-\alpha(P-1),\qquad E_G=S_G/P.
\]

The cube-root rule above concerns scaling of the parallel arithmetic kernel; it does not prove that `α` remains fixed or that wall-clock time is constant. The serial fraction `f` of a one-processor baseline is not generally equal to this `α`. For a matched workload in the two-component ideal model, `f = α/[α+P(1−α)]`.

Distributed GEMM must move matrix blocks and can require replicated panels, buffers, collective communication, and redistribution. The `M/P` entries are ideal averages for storage without replication, not guarantees for every algorithm or processor layout. At increasing `P`, local memory decreases under this scaling, potentially reducing arithmetic intensity and making communication relatively more important.

## Arithmetic validation log

Python integer arithmetic verified the illustrative rows using `N = 1000q`, `P = q³`, `W = 2N³`, and `M = 24N²`.

```text
P=1:   N=1000, W/W1=1,   M/M1=1,  M/(P*M1)=1
P=8:   N=2000, W/W1=8,   M/M1=4,  M/(P*M1)=1/2
P=64:  N=4000, W/W1=64,  M/M1=16, M/(P*M1)=1/4
P=512: N=8000, W/W1=512, M/M1=64, M/(P*M1)=1/8
All rows: W/P = 2000000000 leading-order FLOPs. PASS
```

**Conclusion:** use `N=N₁P^(1/3)`, not `N₁sqrt(P)`, for constant per-processor classical GEMM work. Total work grows linearly, but total three-matrix memory grows only as `P^(2/3)` under the stated idealization.