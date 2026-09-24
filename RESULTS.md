# Lab Results — Shared-Memory Concurrency & OpenMP Paradigms

Machine: **AMD Ryzen 5 7640HS, 6 physical cores / 12 logical threads, L2 6 MB, L3 16 MB**,
Windows 11, OpenJDK 21.0.2. All times are wall-clock (`System.nanoTime()`), after JIT warm-up.
Raw data: `labN/results/`, figures: `plots/`.

---

## Lab 1 — Fork-Join Model, Team Creation, Thread Scoping

### Task 1.1 — Non-determinism (10 consecutive runs, team of 4)
Full log: `lab1/results/task1_1_nondeterminism.txt`. Two distinct print orders were observed
across runs — `(3, 0, 2, 1)` and `(3, 0, 1, 2)` — and the logical rank order never matched the
creation order `0,1,2,3`.

**Q1.1:** Print order is non-sequential because after the fork all threads are runnable and the
**OS scheduler** (the Windows thread dispatcher, driven by core availability, timer interrupts
and priority boosts) decides which thread reaches `println` first. There is no happens-before
edge between the threads' print statements, so any interleaving is a legal execution.

### Task 1.2 — Oversubscription sweep (fork/join cost, 10 trials each)

| P | avg ms | min ms |
|---|--------|--------|
| 1 | 0.371 | 0.342 |
| 2 | 0.624 | 0.459 |
| 4 | 0.911 | 0.506 |
| 8 | 0.802 | 0.558 |
| 16 | 1.151 | 0.910 |
| 32 | 1.366 | 1.010 |
| 64 | 1.662 | 1.265 |

Fork/join of an empty team costs ~0.37 ms at P=1 and grows to ~1.67 ms at P=64 (~4.5×):
pool creation, worker spawning and barrier wake-ups scale with team size.

### Task 1.3 — CPU saturation (10,000,000 `sqrt` per thread)

| P | time (ms) | throughput (Msqrt/s) |
|---|-----------|----------------------|
| 1 | 39.4 | 253.8 |
| 2 | 37.3 | 536.3 |
| 4 | 44.8 | 893.4 |
| 8 | 69.3 | 1153.8 |
| 16 | 119.0 | 1344.5 |
| 32 | 212.3 | 1507.3 |
| 64 | 409.1 | 1564.6 |

**Q1.2:** Each thread carries a fixed 10M-op workload, so ideal wall time would stay ~37 ms.
Beyond the 6 physical cores (and even past 12 SMT threads) throughput flattens at ~1.5 Gsqrt/s
while wall time grows linearly with P: runnable threads exceed hardware contexts, so the OS
**time-slices** them. Each switch costs a **context switch** (saving/restoring registers, stack
pointer, program counter), and migrating a thread between cores causes **cache thrashing** — its
working set is cold in the new core's L1/L2.

**Q1.3:** The implicit barrier at the end of a parallel region guarantees that no thread's work
is still in flight when the master continues. Without it, later code could read half-written
shared data (a race) or consume results of computations that have not finished — classic
read-before-write hazards.

**Q1.4:** A **hardware thread** (Hyper-Threading/SMT) is a physical set of architectural
registers sharing one core's execution units — true simultaneous execution, visible to the OS as
a logical CPU. An **OS kernel thread** is a schedulable software entity with its own stack and
context, multiplexed onto hardware threads by the OS dispatcher. A **green/virtual thread** is
multiplexed by the language runtime onto a small pool of kernel threads — much cheaper to create
and block, but invisible to the OS scheduler.

---

## Lab 2 — Numerical Integration (Pi) & Parallel Reductions

### Task 2.1 — Naive race (N = 100M)

| P | computed Pi | abs. error | time (ms) |
|---|-------------|------------|-----------|
| 1 | 3.141592653590 | 6.33e-13 | 1392.5 |
| 2 | 1.617898548774 | 1.524 | 1124.3 |
| 4 | 0.572894215305 | 2.569 | 285.8 |
| 8 | 0.424120391960 | 2.717 | 145.2 |

With P=1 the sum is exact; with more threads the result collapses — up to **86% of the value is
lost** at P=8. More threads ⇒ more simultaneous read-modify-write interleavings ⇒ more lost updates.

**Q2.1:** `sharedSum[0] += term` compiles to a 3-step RMW: `LOAD` the double into a register,
`ADD` the term, `STORE` it back, and the sequence is **not atomic**. If thread A loads the value
and is preempted before storing while thread B completes a full LOAD–ADD–STORE, A's later store
overwrites B's contribution — one update is *lost*. Over millions of iterations under contention,
most increments are lost.

### Task 2.2 — Critical section (N = 1M), serial baseline 3.8 ms

| P | computed Pi | abs. error | time (ms) | overhead vs. serial |
|---|-------------|------------|-----------|---------------------|
| 1 | 3.141592653590 | 2.9e-14 | 8.7 | +126.8% |
| 2 | 3.141592653590 | 7.9e-14 | 12.5 | +228.0% |
| 4 | 3.141592653590 | 4.9e-14 | 11.9 | +211.8% |
| 8 | 3.141592653590 | 9.9e-14 | 13.2 | +245.1% |

Results are now numerically correct, but a per-step `synchronized` block makes the parallel
version **2.2–3.5× slower than serial** — the lock serializes essentially the whole loop.

**Q2.2:** A centralized critical section serializes P contributors: total synchronization cost
**O(P)** (queueing behind one monitor). A **reduction tree** keeps per-thread partial sums in
private accumulators and merges them pairwise in log₂(P) levels — cost **O(log P)** — while the P
partial sums themselves are computed concurrently with zero contention.

### Task 2.3/2.4 — Strong scaling of the parallel reduction (N = 100M, 5 trials)

| P | avg time (ms) | speedup S(P) | efficiency E(P) | abs. error |
|---|---------------|--------------|------------------|------------|
| 1 | 1067.84 | 1.000 | 1.000 | 4.4e-16 |
| 2 | 547.19 | 1.952 | 0.976 | 4.4e-16 |
| 4 | 260.19 | 4.104 | 1.026 | 4.4e-16 |
| 8 | 132.24 | 8.075 | 1.009 | 4.4e-16 |
| 16 | 82.93 | 12.877 | 0.805 | 4.4e-16 |

Near-linear scaling to P=8 (SMT helps beyond 6 physical cores for this FPU-light loop), then
efficiency drops to 0.80 at P=16 (oversubscription of 12 logical CPUs). See
`plots/lab2_speedup.png`.

**Q2.3:** Amdahl: with serial fraction s = 0.05, S_max = 1/s = **20×** even with infinitely many
processors. If measured speedup flattens earlier, secondary causes are memory-bandwidth
saturation, SMT siblings sharing one core's execution units, all-core turbo/thermal frequency
reduction, and scheduling overhead that grows with P.

**Q2.4:** On x86-64 an atomic RMW such as `LOCK XADD` / `LOCK CMPXCHG` does not lock the whole
bus: the core's cache controller first acquires the target cache line in **Modified (MESI)**
state — invalidating all other copies via **bus snooping** — then performs the update while
holding off other snoops for that line, and releases it afterwards. ARM implements the same
semantic with **Load-Linked/Store-Conditional** (`LDAXR`/`STLXR`): the conditional store succeeds
only if no other core touched the line since the load, otherwise the operation retries.

---

## Lab 3 — Work-Sharing & Loop Scheduling (Mandelbrot 1920×1080, max_iter = 1000)

### Task 3.2 — 4×4 sweep, mean of 3 runs (ms). Raw data: `lab3/results/task3_matrix.csv`

| Threads | Static (block) | Dyn C=1 | Dyn C=16 | Dyn C=64 | Dyn C=256 |
|---------|----------------|---------|----------|----------|-----------|
| 2 | 430 | 431 | 429 | 451 | 524 |
| 4 | 432 | 224 | 223 | 238 | 538 |
| 8 | 363 | 119 | 119 | 214 | 522 |
| 16 | 244 | 103 | 97 | 215 | 537 |

Heatmap + comparison chart: `plots/lab3_scheduling.png`.

Key findings:
- **Static scheduling barely scales from P=2 to P=4 (430 → 432 ms)** because one thread gets the
  block containing the set's interior (rows through the main cardioid) and becomes a straggler.
- **Dynamic with C=1/16 is ~1.9× faster than static at the same P** (e.g. P=8: 119 vs 363 ms).
- **C=256 is pathological**: 1080 rows / 256 ≈ only 5 chunks exist, so at P≥4 some threads get no
  chunk at all while one thread grinds a huge block — worse than static (~525–545 ms at any P).
- C=1 and C=16 are nearly equal here (rows are heavy enough that 1080 atomic dispenses are
  cheap), but C=64/256 show the coarse-granularity penalty.

### Task 3.4 — Load imbalance = (Max − Min) / Average per-thread work

| Scheduler | P | C | imbalance |
|-----------|---|---|-----------|
| static | 4 | — | **1.974** |
| dynamic | 4 | 1 | 0.025 |
| dynamic | 4 | 64 | 0.120 |
| static | 8 | — | **3.262** |
| dynamic | 8 | 1 | 0.071 |
| dynamic | 8 | 64 | 1.883 |

Static per-thread work at P=4: `[0.94M; 100.1M; 101.2M; 0.94M]` — the two middle ranks compute
~100× more escape iterations than the outer ranks. Dynamic C=1 equalizes work to within 2.5%.

**Q3.1:** With C=1 every chunk claim is one atomic `getAndAdd` on the shared dispenser. The
contested hardware resource is the **single cache line holding the queue counter**: it ping-pongs
between cores in Modified state on every claim (a MESI invalidation round-trip per row). When
chunk work is small, this coherency traffic approaches the cost of the work itself.

**Q3.2:** Under static block partitioning, the stragglers are the ranks whose row interval
crosses the **main cardioid and the period-2 bulb near the horizontal center of the complex
plane** (|Im| small): those pixels never escape, costing the full 1000 iterations. Here those are
the middle ranks (1 and 2 of 4; 3 and 4 of 8). Edge ranks render mostly exterior pixels that
escape after a handful of iterations.

**Q3.3:** `schedule(guided, chunk)` dispenses chunks whose size is proportional to the *remaining*
iterations divided by P (roughly `ceil(remaining / P)`), never smaller than `chunk`. Early
iterations go out in huge blocks — few dispenser operations, like static — while the tail goes
out in ever-smaller pieces, so no thread is stuck with a large final block while others idle. It
combines static's low overhead with dynamic's balance.

**Q3.4:** Rule of thumb — **Static** when iteration cost is uniform and N ≫ P; **Dynamic** with a
moderate chunk when cost is unpredictable/coarse (Mandelbrot, adaptive refinement); **Guided**
when cost is irregular *and* the iteration count is so large that fine dynamic dispatching
overhead would be measurable. Always size the chunk so one chunk costs at least a few
microseconds of work.

---

## Lab 4 — Cache Coherency & False Sharing (100M `volatile` increments/thread, mean of 2 runs)

Raw data: `lab4/results/task4_scaling.csv`, plot: `plots/lab4_false_sharing.png`.

| P | Unpadded (ms) | Padded 64B stride (ms) | Thread-local register (ms) |
|---|---------------|-------------------------|----------------------------|
| 1 | 88.5 | 81.5 | 25.5* |
| 2 | 460.5 | 185.5 | 4.5 |
| 4 | 1451.5 | 175.5 | 4.0 |
| 8 | 2021.5 | 331.5 | 7.0 |
| 16 | 3428.0 | 283.0 | 13.5 |

*first-trial JIT residue; steady-state single-thread local time is ~3 ms.

- **Unpadded gets *slower* as threads are added** — 88 ms at P=1 but 3428 ms at P=16 (~39× worse),
  because the counters sit in one 64-byte line that bounces between cores on every increment.
- **Padding (64-byte stride) recovers nearly flat scaling** (~180–330 ms regardless of P).
- **Thread-local register accumulation is the real fix**: 3–14 ms total, i.e. **~250× faster**
  than unpadded at P=16, because increments never leave the CPU register until one final store.

**Q4.1:** The line holds counters c0..c3 (8 bytes each). Trace for alternating writes by Core 0
(c0) and Core 1 (c1): Core 0 loads the line → **Exclusive**, writes c0 → **Modified**. Core 1
wants to write c1: it snoops the line, Core 0 flushes it and drops to **Invalid**; Core 1 loads
→ **Modified**, writes c1. Core 0's next write finds its copy **Invalid** → snoop/flush from
Core 1, and so on: the line alternates M→I→M→I on every increment, each transition a costly
coherency transaction instead of an L1 hit.

**Q4.2:** More cores means more contenders for the *same physical cache line*; the bottleneck is
the **inter-core coherency fabric** (snoop bus / Infinity Fabric + L3 as the point of coherence).
Every increment requires a line transfer of tens of ns; serializing 100M × P increments through
one line grows with P, so the machine performs *worse* than one uncontended core.

---

## Lab 5 — Recursive Task-Based Parallelism (Merge Sort, N = 5,000,000)

Raw output: `lab5/results/task5_results.txt`, plot: `plots/lab5_cutoff.png`.

**Task 5.1:** Parallel sort output verified ascending: `true` (re-asserted for every cutoff).

Sequential baseline (T1, `Arrays.sort`): **553.4 ms**.

### Task 5.2 — Cutoff sweep (3 trials each)

| Cutoff K | avg (ms) | min (ms) | speedup vs. sequential |
|----------|----------|----------|------------------------|
| 10 | 184.9 | 136.2 | 2.99× |
| 100 | 142.7 | 138.0 | 3.88× |
| 1,000 | 135.1 | 129.7 | 4.10× |
| 10,000 | 137.6 | 124.3 | 4.02× |
| **50,000** | **133.2** | **122.7** | **4.16×** |
| 100,000 | 136.6 | 129.1 | 4.05× |

Optimum at K ≈ 10³–10⁵ with a shallow trough; K = 10 is ~39% slower than the optimum.

### Task 5.3 — Work–Span model
- Work T1 = **O(N log N)** (all comparisons and merge passes). Measured T1 ≈ 553 ms.
- Span T∞ = **O(N)**: recursion depth is O(log N), but the *root merge* alone is a sequential
  O(N) pass over all N elements and dominates the critical path.
- Theoretical parallelism P = T1/T∞ = **O(log N) ≈ 22** for N = 5·10⁶; empirical speedup 4.16× on
  6 cores is well below that bound — the run is hardware-limited, not span-limited.

**Q5.1:** With K = 1 the recursion creates ~2N task objects: for N = 5M that is ~10⁷
`RecursiveAction` allocations, each with object header, fields, deque slot and stack frame —
allocation churn, GC pressure and enqueue/dequeue work dwarf the single comparison each task
performs. Scheduling overhead swamps throughput (visible already at K = 10: 185 ms vs 133 ms).

**Q5.2:** Each worker owns a **local double-ended queue (deque)**. It *pushes* new tasks and
*pops* its next task from the **head** (LIFO — cache-hot, uncontended). An idle worker *steals*
from the **tail** of a random victim's deque (FIFO — the oldest, largest chunk of work), which
minimizes steal frequency and keeps the common case lock-free.

**Q5.3:** The sequential merge keeps O(N) work on the critical path, capping parallelism at
O(log N) regardless of how many cores sort the halves. The merge can itself be parallelized by
**co-ranking**: each of P workers binary-searches its rank boundaries in both sorted halves and
merges a disjoint output slice, reducing merge span to O(N/P + log N).

**Q5.4:** `#pragma omp for` needs a canonical loop with a known trip count — perfect for dense
arrays and stencil loops. `#pragma omp task` handles **irregular, dynamically generated** work:
divide-and-conquer algorithms, tree/graph traversal, pointer chasing, producer–consumer
pipelines — domains where the amount of work is discovered at run time and loop partitioning is
impossible a priori.

---

## Conclusions (Section V)

1. Thread teams are cheap but not free: fork/join cost grows with team size, and CPU-bound
   throughput saturates at the logical-CPU count (12) — oversubscription only adds context-switch
   overhead.
2. Synchronization granularity dominates correctness *and* performance: a naive shared
   accumulator loses up to 86% of updates; a per-step critical section is 2–3.5× slower than
   serial; a tree reduction scales near-linearly (8.1× on 8 threads).
3. For non-uniform loops, dynamic self-scheduling beat static partitioning ~1.9×, but chunk size
   must be tuned — too coarse (C=256) is worse than static.
4. Memory layout can matter more than the algorithm: false sharing made 16 threads 39× slower
   than one; 64-byte padding fixed scaling, and thread-local registers removed coherency traffic
   entirely (~250× vs. unpadded at P=16).
5. Recursive tasking needs a granularity cutoff: an optimum near K = 50,000 gave 4.16× on 6
   cores, while K = 10 wasted ~40% on task-management overhead.
