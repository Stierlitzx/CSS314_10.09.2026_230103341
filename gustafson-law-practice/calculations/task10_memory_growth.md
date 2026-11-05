# Task 10 — Memory-limited growth of a three-dimensional grid

**Student ID:** 230103341

## Inputs, missing information, and assumptions

Each node has 32 GB; compare 64 and 512 nodes. Let `N` be the **global linear dimension** of a cubic grid, so the number of cells is `N³`, not `N`. Let `b` be the total resident bytes per owned cell, including every field and time level that must coexist.

There is **no unique numerical maximum N without b**. A floating-point precision alone does not determine `b`: multiple solution arrays, material coefficients, scratch space, and metadata may all occupy memory. The following bound assumes perfectly distributable storage, no replicated arrays, no halo/ghost cells, no runtime/OS reservation, and all advertised memory available. It is therefore an aggregate-memory upper bound, not a guarantee that a particular decomposition fits.

Use decimal units here: `1 GB = 10^9 bytes`. In contrast, `1 GiB = 2^30 bytes`; 32 GB is 29.8023223876953125 GiB, while 32 GiB is 34.359738368 GB. Replacing GB by GiB changes the answer and must be stated explicitly.

## General answer

For `p` nodes, available aggregate memory is `M(p) = 32 × 10^9 p` bytes. Thus

\[
bN^3\le 32\times10^9p,\qquad
\boxed{N_{\max}(p,b)=\left\lfloor\left(\frac{32\times10^9p}{b}\right)^{1/3}\right\rfloor}.
\]

In particular,

\[
N_{64}=\left\lfloor(2{,}048{,}000{,}000{,}000/b)^{1/3}\right\rfloor,
\quad
N_{512}=\left\lfloor(16{,}384{,}000{,}000{,}000/b)^{1/3}\right\rfloor.
\]

Before integer rounding, increasing the node count by eight multiplies cell capacity by eight and **linear dimension by two**, since `8^(1/3) = 2`. If work is proportional to cell count (fixed number of stencil steps), total work can grow eightfold. This does not imply that runtime stays constant: communication, bandwidth, and convergence can change.

## Explicitly illustrative numerical cases — not inferred data

* `b = 8 bytes/cell`: for example, one binary64 scalar array only. Most iterative solvers need more than this.
* `b = 64 bytes/cell`: for example, eight resident binary64 values per owned cell, still excluding halos and other overhead.

| Illustrative b (bytes/cell) | Nodes p | Substitution | Global aggregate bound N | Global cell count N³ |
|---:|---:|---|---:|---:|
| 8 | 64 | `floor((2,048,000,000,000 / 8)^(1/3))` | 6,349 | 255,926,926,549 |
| 8 | 512 | `floor((16,384,000,000,000 / 8)^(1/3))` | 12,699 | 2,047,899,168,099 |
| 64 | 64 | `floor((2,048,000,000,000 / 64)^(1/3))` | 3,174 | 31,975,752,024 |
| 64 | 512 | `floor((16,384,000,000,000 / 64)^(1/3))` | 6,349 | 255,926,926,549 |

Integer floors explain why the listed dimensions do not always double exactly.

### Exact capacity checks (bytes)

The two inequalities `bN³ ≤ capacity < b(N+1)³` establish each floor without relying on floating-point cube-root rounding.

| b | p | bN³ | Capacity | b(N+1)³ | Unused aggregate bytes |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 2,047,415,412,392 | 2,048,000,000,000 | 2,048,383,000,000 | 584,587,608 |
| 8 | 512 | 16,383,193,344,792 | 16,384,000,000,000 | 16,387,064,000,000 | 806,655,208 |
| 64 | 64 | 2,046,448,129,536 | 2,048,000,000,000 | 2,048,383,000,000 | 1,551,870,464 |
| 64 | 512 | 16,379,323,299,136 | 16,384,000,000,000 | 16,387,064,000,000 | 4,676,700,864 |

## Local partition shape and non-ideal storage

Aggregate free memory is not automatically usable by an overloaded rank. If `p = q³` and the domain is split into `q × q × q` blocks, a nondivisible `N` gives a largest block of `ceil(N/q)³` cells. In each illustrative aggregate maximum above, this largest block would require **32,036,235,776 bytes**, exceeding 32 GB, even though the aggregate bound passes. A different distribution or a smaller dimension is needed.

For equal cubic blocks with `N = qm` and no overhead, a safe largest local dimension is

\[
m=\left\lfloor(32\times10^9/b)^{1/3}\right\rfloor,\quad N=qm.
\]

| b | Local m | Local bytes b m³ | Next local cube b(m+1)³ | Global N at 64 nodes (q=4) | Global N at 512 nodes (q=8) |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,587 | 31,975,752,024 | 32,036,235,776 | 6,348 | 12,696 |
| 64 | 793 | 31,915,344,448 | 32,036,235,776 | 3,172 | 6,344 |

These are maxima **under that equal-cube restriction**, not replacements for the general aggregate bound. Slabs, pencils, uneven blocks, and the ranks-per-node layout have different local shapes and halo footprints.

With a halo of width `h`, a simple full padded-cube storage model becomes `b(m+2h)³` bytes per rank rather than `bm³`. A more realistic node constraint is

\[
\sum_{r\text{ on node}} b(m_{x,r}+2h)(m_{y,r}+2h)(m_{z,r}+2h)
+M_{\rm replicated}+M_{\rm scratch}+M_{\rm runtime}
\le M_{\rm usable}.
\]

Replicated global arrays do not benefit from aggregate capacity in the same way as distributed owned cells. Halo duplication, alignment, buffers, OS reserve, and replication must be specified before claiming an attainable maximum.

## Arithmetic validation log

Python integer arithmetic checked all four aggregate inequalities and both local inequalities. Decimal results shown above are model calculations, not measured allocations.

```text
b=8,  p=64:  2047415412392 <= 2048000000000  < 2048383000000  PASS
b=8,  p=512: 16383193344792 <= 16384000000000 < 16387064000000 PASS
b=64, p=64:  2046448129536 <= 2048000000000  < 2048383000000  PASS
b=64, p=512: 16379323299136 <= 16384000000000 < 16387064000000 PASS
b=8:  local m=1587; 31975752024 <= 32000000000 < 32036235776 PASS
b=64: local m=793;  31915344448 <= 32000000000 < 32036235776 PASS
```

**Conclusion:** the correct unconditional answer is the formula in `b`, together with the storage assumptions. The `b=8` and `b=64` tables are explicitly illustrative, not an invented unique solution.