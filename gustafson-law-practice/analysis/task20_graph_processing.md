# Task 20 — Graph processing: useful work is not uniformly distributed

**Student ID:** 230103341

## 1. Define the graph and the workload

Let \(n\) be vertex count and \(m\) edge count. A full adjacency-list BFS has \(O(n+m)\) sequential work over the reachable graph. One conventional sparse PageRank iteration performs \(O(n+m)\) work, including dangling-node handling and vector updates. Neither expression alone guarantees \((n+m)/P\) elapsed time: irregular memory access, edge cuts, synchronization and skewed degree distributions matter.

Power-law-like graphs can contain hubs whose degrees greatly exceed the average. Such a degree distribution does not by itself specify the partition edge cut, diameter or network traffic. Conversely, dense graphs have \(m=\Theta(n^2)\); their greater arithmetic volume can sometimes amortize coordination, but graph traversal is not automatically a compute-dense matrix multiplication. The following comparisons are analytical models, not benchmark results.

## 2. BFS: frontier size and edge balance dominate

For level-synchronous BFS, let \(L\) be the number of reached levels, \(e_{\ell,r}\) the edges processed by worker \(r\) at level \(\ell\), and \(C_\ell\) exposed communication plus termination/barrier cost. A useful model is

\[
T_P\approx T_{\rm setup}
+\sum_{\ell=0}^{L-1}\left[k\max_r e_{\ell,r}+C_\ell(P)\right].
\]

Perfectly balanced edge work would replace the maximum by \(\sum_r e_{\ell,r}/P\). The difference is imbalance. A partition with equal numbers of vertices can still give one worker most of a frontier's edges because it owns a hub. Splitting hub adjacency lists across workers, using vertex cuts or redistributing frontier work can improve balance, but may add replication, messages and visited-state coordination.

Small initial and final frontiers expose little concurrency even if total graph size is large. Cross-partition neighbor discovery requires exchanging candidate vertices and suppressing duplicates. More processors may create more remote edges or smaller messages. These costs are not necessarily one fixed serial setup fraction; they recur at each level and depend on the selected source.

For illustration, in a star with one hub owned by one worker and an **unsplit** adjacency list, a vertex-balanced partition can leave \(\Theta(m)\) edge work on that worker in the hub's expansion level. Increasing \(P\) does not divide that level's critical path by \(P\). This is an ownership-policy counterexample, not a theorem that every star BFS is poorly parallelizable.

## 3. PageRank: regular iterations over irregular data

With damping \(0<d<1\), a standard PageRank update distributes each non-dangling vertex's rank along outgoing edges, adds teleportation, and redistributes dangling mass. A sparse implementation is a sparse matrix–vector operation plus global sums, exchanges and potentially a convergence test. The original PageRank report describes the ranking method and its iterative computation. [PR]

For \(K\) iterations, write

\[
T_P\approx T_{\rm setup}+\sum_{j=1}^{K}
\left[k\max_r m_{j,r}+C_{\rm exchange}(P,n,m)
+C_{\rm reduce}(P)+I_j\right].
\]

Here \(m_{j,r}\) denotes assigned edge work, while \(I_j\) covers remaining effects not already included in edge imbalance. A fixed-size tree reduction can have logarithmic depth, but exchanging rank contributions can dominate it. Edge-balanced partitions improve arithmetic balance without guaranteeing low communication. Hubs cause update contention in push implementations or large adjacency scans in pull implementations.

State whether \(K\) is fixed or determined by a specified residual norm and tolerance. Enlarging a graph may change the convergence behavior; assuming a fixed number of iterations while claiming equal solution accuracy can make a scaling comparison misleading. Dangling vertices, normalization and residual checks are part of correctness, not optional benchmark shortcuts.

## 4. Sparse/power-law versus dense growth paths

| Property | Sparse graph family, \(m=\Theta(n)\) | Dense graph family, \(m=\Theta(n^2)\) |
|---|---|---|
| Ideal work per complete edge pass | \(\Theta(n)\) | \(\Theta(n^2)\) |
| Ideal constant-work-per-worker growth | \(n=\Theta(P)\) | \(n=\Theta(\sqrt P)\) |
| Typical scaling risk | Hubs, irregular access, remote edges | Data volume, bandwidth, synchronization |
| Required extra specification | Degree/partition/source distribution | Representation and traversal/update algorithm |

The sparse row assumes bounded average degree along the family; “power law” alone does not guarantee that assumption. BFS may inspect different numbers of edges with direction-optimizing versus ordinary push traversal, especially on dense graphs. Thus count validated useful graph operations under a declared algorithm, rather than assigning the same work to all implementations solely from \(m\).

## 5. Use Graph500 precisely

The official historical **Graph 500 Benchmark 1 (Search), specification v1.2 (2011)** defines a Kronecker-style graph generator with \(n=2^{\mathrm{SCALE}}\) and an input edge-tuple count \(\mathrm{edgefactor}\times n\). It times graph construction and BFS separately, uses multiple search roots, validates BFS parent trees, and reports traversal rate in TEPS with a harmonic-mean summary. [G500] Duplicate edges and self-loops mean input tuple count must not be casually equated with the count of unique stored undirected edges. Its TEPS numerator follows the specification's reached-component edge accounting, not an arbitrary software counter.

At fixed edgefactor, increasing SCALE by one doubles vertices and generated tuples. This is a sparse input-growth path, not a fixed-size speedup experiment. Construction time is reported separately; BFS validation is outside the timed BFS call in the reference driver. Cite the exact version and timing boundaries rather than claiming BFS-only TEPS measures end-to-end ingestion or PageRank. This historical search specification does **not** define a PageRank benchmark.

For a defensible study, record generator parameters, graph representation, partition, root set, precision/tolerance, processed edges, memory usage and communication. Test both a fixed graph and a documented growing family. Reconstruct scaled speedup using the serial time for the same enlarged graph and algorithm, explicitly labeling modeled references when the graph cannot fit on one node. A high TEPS result on one graph family is not evidence of a universal constant serial fraction or universal graph scalability.

## Primary sources

- [G500] Graph500 Steering Committee, official Search specification v1.2: <https://github.com/graph500/graph500/blob/master/Graph500.org> ([raw specification](https://raw.githubusercontent.com/graph500/graph500/master/Graph500.org)); [reference repository](https://github.com/graph500/graph500). Version is stated explicitly because later suites/rules may differ.
- [PR] L. Page, S. Brin, R. Motwani and T. Winograd, “The PageRank Citation Ranking: Bringing Order to the Web,” Stanford technical report (1999): <http://ilpubs.stanford.edu:8090/422/>. The repository endpoint was unavailable during preparation; graph timing formulas here are independent analytical models, not attributed empirical findings from this report.
