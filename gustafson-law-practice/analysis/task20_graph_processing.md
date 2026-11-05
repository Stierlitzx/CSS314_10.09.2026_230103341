# Task 20 — Graph processing: uneven useful work

**Student ID:** 230103341

## My workload and timing models

I use n vertices, m edges and P workers. Reachable-graph adjacency-list BFS and one sparse PageRank iteration each need O(n+m) work, not guaranteed O((n+m)/P) time. Hubs, irregular access, cuts and synchronization matter. Power-law degrees determine neither traffic nor diameter; dense traversal is not necessarily compute-dense matrix multiplication. These are analytical models, not benchmarks.

For level-synchronous BFS, L is reached levels, e[level,r] is worker r’s processed edges, k is time/edge, and C[level] is exposed communication plus barrier/termination cost:

```text
Tₚ ≈ T_setup + sum over levels 0…L−1 of
     (k × max_r e[level,r] + C[level](P))
Balanced edge count per worker = (sum_r e[level,r])/P
```

Equal vertex counts do not balance frontier edges. In a star with one worker owning the hub’s unsplit adjacency list, its expansion retains Θ(m) critical-path work despite increasing P. This is an ownership counterexample, not inherent star-BFS failure. Splitting adjacency, vertex cuts or frontier redistribution can help but add replication/messages/visited-state coordination. Small initial/final frontiers limit concurrency; remote discoveries require exchange and duplicate suppression. Costs recur by level and source, not as one constant serial fraction.

For PageRank damping 0 < d < 1, each iteration distributes outgoing rank, teleportation and dangling mass [PR]. With K iterations and assigned edge work m[j,r]:

```text
Tₚ ≈ T_setup + sum over j=1…K of
     (k × max_r m[j,r] + C_exchange(P,n,m) + C_reduce(P) + I[j])
```

Tₚ is elapsed time; T_setup is setup, C terms are exposed communication/reduction, and I[j] covers remaining effects, excluding already-counted edge imbalance. Tree reduction can have logarithmic depth while rank exchange dominates. Edge balance does not guarantee low traffic; hubs create push contention or long pull scans. I distinguish fixed K from residual-norm/tolerance stopping: larger graphs can converge differently. Dangling handling, normalization and residual checks remain correctness requirements.

## My growth-path comparison

| Complete edge pass | Sparse family | Dense family |
|---|---|---|
| Edges/work | m = Θ(n); work Θ(n) | m = Θ(n²); work Θ(n²) |
| Ideal constant work/worker | n = Θ(P) | n = Θ(√P) |
| Main risks | Hubs, remote edges, irregular access | Volume, bandwidth, synchronization |

Sparse growth assumes bounded average degree—not guaranteed by “power law.” I specify degree/partition/source distributions, representation and algorithm. Direction-optimizing versus push BFS can inspect different edge counts, especially on dense graphs; I count validated operations, not just m.

## Graph500 scope and evidence

I use historical **Graph 500 Benchmark 1 (Search), v1.2 (2011)** [G500]: Kronecker-style generation with n = 2^SCALE and edgefactor × n input tuples. At fixed edgefactor, SCALE + 1 doubles both: sparse growth, not fixed-work speedup. Duplicates/self-loops prevent equating tuples with unique undirected stored edges. TEPS follows reached-component edge accounting and a harmonic-mean summary across search roots.

Construction and BFS are timed separately; parent-tree validation is outside timed BFS in the reference driver. This specification defines neither end-to-end ingestion TEPS nor PageRank benchmarking. My study plan records generator, representation, partition, roots, precision/tolerance, processed edges, memory and communication for fixed and growing graphs. Scaled speedup needs the same enlarged graph’s serial runtime, explicitly modeled if it cannot fit one node. One family’s TEPS cannot establish universal scalability or constant serial fraction.

- [G500] Official [Search specification v1.2](https://github.com/graph500/graph500/blob/master/Graph500.org), [raw specification](https://raw.githubusercontent.com/graph500/graph500/master/Graph500.org), [reference repository](https://github.com/graph500/graph500). Later rules may differ.
- [PR] Page, Brin, Motwani and Winograd, “The PageRank Citation Ranking: Bringing Order to the Web,” [Stanford report (1999)](http://ilpubs.stanford.edu:8090/422/). The endpoint was unavailable during preparation; my timing models are not attributed empirical findings.