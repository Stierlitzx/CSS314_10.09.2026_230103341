# Task 10 — Memory-limited three-dimensional grid growth

**Student ID:** 230103341

## My capacity model

I compare 64 and 512 nodes with 32 GB each. N is the global cubic-grid edge; b is total resident bytes per owned cell, including all coexisting fields and time levels. **Without b, I cannot give a unique maximum N.** Precision alone does not specify storage.

I use decimal GB and initially assume perfectly distributed storage, no halos, replication, scratch/runtime overhead, or OS reserve, and all advertised memory available.

```text
1 GB = 10^9 bytes; 1 GiB = 2^30 bytes
32 GB = 29.8023223876953125 GiB
32 GiB = 34.359738368 GB

For p nodes: M(p) = 32 × 10^9 × p bytes
bN³ ≤ M(p)
N_max(p,b) = floor((M(p)/b)^(1/3))
N_64 = floor((2,048,000,000,000/b)^(1/3))
N_512 = floor((16,384,000,000,000/b)^(1/3))
```

Before flooring, eight times as many nodes allow eight times the cells but only twice the edge length. For fixed stencil-step count, work can grow eightfold; constant runtime is not guaranteed.

## My illustrative results

I retain two examples, not inferred storage data: b = 8 for one binary64 scalar array, and b = 64 for eight resident binary64 values. Both exclude overhead.

| b (bytes/cell) | Nodes p | Substitution | Aggregate bound N | Cells N³ |
|---:|---:|---|---:|---:|
| 8 | 64 | floor((2,048,000,000,000 / 8)^(1/3)) | 6,349 | 255,926,926,549 |
| 8 | 512 | floor((16,384,000,000,000 / 8)^(1/3)) | 12,699 | 2,047,899,168,099 |
| 64 | 64 | floor((2,048,000,000,000 / 64)^(1/3)) | 3,174 | 31,975,752,024 |
| 64 | 512 | floor((16,384,000,000,000 / 64)^(1/3)) | 6,349 | 255,926,926,549 |

I verify each floor with exact integer byte counts:

```text
bN³ ≤ capacity < b(N + 1)³
```

| b | p | bN³ | Capacity | b(N+1)³ | Unused bytes |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 2,047,415,412,392 | 2,048,000,000,000 | 2,048,383,000,000 | 584,587,608 |
| 8 | 512 | 16,383,193,344,792 | 16,384,000,000,000 | 16,387,064,000,000 | 806,655,208 |
| 64 | 64 | 2,046,448,129,536 | 2,048,000,000,000 | 2,048,383,000,000 | 1,551,870,464 |
| 64 | 512 | 16,379,323,299,136 | 16,384,000,000,000 | 16,387,064,000,000 | 4,676,700,864 |

## My local-memory check

For a q × q × q node decomposition, the largest uneven block at every aggregate maximum above needs **32,036,235,776 bytes**, exceeding 32 GB. Aggregate fit is therefore insufficient. Under equal cubic blocks without overhead:

```text
p = q³; largest uneven block = ceil(N/q)³ cells
Local edge m = floor((32 × 10^9/b)^(1/3)); global N = qm
```

| b | Local m | Local bytes bm³ | Next cube b(m+1)³ | N at 64 nodes (q=4) | N at 512 nodes (q=8) |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,587 | 31,975,752,024 | 32,036,235,776 | 6,348 | 12,696 |
| 64 | 793 | 31,915,344,448 | 32,036,235,776 | 3,172 | 6,344 |

I treat these as maxima only for equal cubes. Slabs, pencils, uneven blocks, and ranks per node change local storage. With halo width h, padded cubic storage is b(m + 2h)³; for local rank edges m_x,r, m_y,r, m_z,r:

```text
sum over ranks r on a node of
  [b(m_x,r + 2h)(m_y,r + 2h)(m_z,r + 2h)]
  + M_replicated + M_scratch + M_runtime ≤ M_usable
```

I checked all aggregate and local inequalities exactly. Halos, alignment, buffers, reserves, and replication must be specified before claiming an attainable maximum.