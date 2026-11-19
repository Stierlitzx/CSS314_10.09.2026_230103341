# ANALYSIS — Cache Line & False Sharing Mechanics (Q1–Q5)

**Ayazgaliyev Aibek · 230103341 · Git `Stierlitzx` · 19 November 2026**

All numbers come from the committed run [`full_run_2026-11-19.txt`](full_run_2026-11-19.txt) and its manifest
[`cache_lab_manifest.json`](cache_lab_manifest.json) (`verification_passed: true`, manifest sha256 `e865527cc49da583…`).

| Run context | Value |
|---|---|
| CPU | AMD Ryzen 5 7640HS (Zen 4, one 6-core CCX, 6 C / 12 T), Windows 11 |
| L1d line size | **64 B** (from `GetLogicalProcessorInformation`) |
| `INSTANCE_SEED` | `16200051438975350361` (`0xe0d224f3c32e7659`) = SHA-256(`Stierlitzx::AMD64::AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD::12`) |
| Threads / pinning | 4 `threading.Thread` workers, `SetThreadAffinityMask` → logical CPUs **0, 2, 4, 6** = 4 different physical cores (SMT pairs `[0,1] [2,3] [4,5] [6,7]…`) |
| Work | 60,000,000 read-modify-write updates per thread, 5 trials per layout, **median** reported |
| Buffer | `uint64`, base address 128-byte aligned (so also 64-byte aligned) |
| Power | battery, 54 %, AC unplugged (laptop power policy active; absolute ns are clock-dependent) |
| Runtime | CPython 3.14.3 (GIL enabled), NumPy 2.5.3, Numba 0.67.0 |

---

## Question 1 — MESI state transitions under `stride = 1`

With `stride = 1`, threads 0–3 update `buffer[0..3]` at byte offsets 0, 8, 16, 24. All four fall in **cache line 0**
(`⌊(base + 8k)/64⌋ = line 0` for k = 0..3; the run prints `lines touched [0]`).

Sequence when Core 0 (logical CPU 0) stores to `buffer[0]` and Core 1 (logical CPU 2) then stores to `buffer[1]`:

| Step | Event | Core 0 copy of line 0 | Core 1 copy of line 0 |
|---|---|---|---|
| 0 | Both cores previously read the line | **S** | **S** |
| 1 | Core 0 `store buffer[0]`: the store retires into Core 0's store buffer and cannot commit until Core 0 owns the line. It issues an upgrade/RFO (invalidate) | S → **M** after the ack | S → **I** (invalidated) |
| 2 | Core 1 `store buffer[1]` (a different address in the same line): its copy is **I**, so it sends a **Read-For-Ownership** | Core 0 snoops the RFO, forwards the dirty line cache-to-cache, **M → I** | I → **E/M**, then the store commits: **M** |
| 3 | Core 0's next `store buffer[0]` (the next loop iteration) | I → RFO → **M** | M → **I** |
| … | Each core's ownership tenure ends whenever another core issues an RFO | ping-pong | ping-pong |

(Zen 4 actually runs a MOESI-family protocol. An RFO still has to invalidate every other copy, so the extra
**O**wned state does not help a store; the M→I→M ping-pong above is what happens.)

**Why the RFO stalls Core 1's store buffer:** x86 is TSO, so stores leave the store buffer and commit to L1 strictly in
program order, and only into a line held in M/E. While Core 1's oldest store waits for its RFO to be answered (a
cache-to-cache transfer through the shared L3 of the CCX, tens of ns), every younger store queues behind it. Our kernel
issues one store per iteration, so the store buffer (64 entries on Zen 4) fills and the front end must stop
allocating, which stalls the core. Loads of `buffer[1]` in the next iteration can still be satisfied by
**store-to-load forwarding** from the store buffer, so the computation itself never waits on memory. The cost is the
draining of stores, which can only happen while the core owns the line.

The run shows this directly: **5.666 ns per update per thread at `stride = 1` vs 0.677 ns for one thread alone**,
i.e. **+4.961 ns per update** of coherence overhead compared with `stride = 8`. That is far below one full cache-to-cache
round trip per store. So each ownership tenure drains a *burst* of queued stores before the line is taken away
again, and the store buffer is what turns 240 M fine-grained stores into far fewer line migrations.

## Question 2 — Measured durations and slowdown

Median of 5 trials, 4 concurrent threads × 60,000,000 updates (240 M updates in total):

| Regime | Median wall (ns) | ns / update / thread | ns / update aggregate (÷240 M) | vs `stride = 8` |
|---|---:|---:|---:|---:|
| Single thread alone (reference) | 40,602,800 | 0.677 | — | 0.961× |
| **`stride = 1`** (8 B, 1 line) | **339,939,400** | **5.666** | 1.416 | **8.043×** |
| `stride = 4` (32 B, 2 lines) | 169,377,500 | 2.823 | 0.706 | 4.008× |
| **`stride = 8`** (64 B, 4 lines) | **42,262,700** | **0.704** | 0.176 | 1.000× |
| `stride = 16` (128 B, 4 lines) | 46,693,300 | 0.778 | 0.195 | 1.105× |
| **Register-private** (Task 4) | **8,506,700** | **0.142** | 0.035 | 0.201× |

* **Slowdown `stride 1 / stride 8` = 339,939,400 / 42,262,700 = 8.043×.** Every trial agrees: stride 1 ranges
  336.4–344.7 ms and stride 8 ranges 41.7–49.7 ms.
* `stride 4` sits almost exactly half way (**4.008×**). Each line is shared by only 2 threads (lines `[0, 1]`), so each
  RFO has one competitor instead of three, and the measured coherence overhead roughly halves:
  +2.119 ns/update vs +4.961 ns/update.
* **`stride 8` vs single thread: 0.704 vs 0.677 ns/update (+4 %).** Once each thread owns its own line, 4 threads run as
  fast as 1. All four pinned workers finished within 40.6–42.1 ms. The kernel scales across cores, which also shows the
  GIL really is released (the manifest check `gil_released: true`).
* **Private register vs `stride 8`: 4.968× faster (0.142 vs 0.704 ns/update).** This difference is the cost of the
  L1 load → add → store → store-forward dependency chain that `stride = 8` still pays every iteration (the clock
  frequency on battery was not recorded, so I do not convert it to cycles). LLVM additionally vectorised the private loop (see Q4). Splitting the regimes:
  **register ≈ 0.14 ns, L1 store-forwarding ≈ 0.70 ns, cross-core coherence ≈ 5.67 ns per update.**
* Context switches (psutil on Windows, voluntary only) stayed between −8 and 294 per trial and do not follow the
  layout. Negative values are an artefact of Windows per-live-thread accounting. Context switches therefore do not
  explain the slowdown: it is hardware coherence traffic, not the OS.
* Every final counter is bit-identical across all layouts, all trials and the private baseline:
  `[1949999939, 1950000037, 1950000048, 1950000014]`.

## Question 3 — L2 spatial prefetcher and 128-byte pairing

Measured: `stride 8` (64 B spacing, lines 0, 1, 2, 3) median **42,262,700 ns**, `stride 16` (128 B spacing, lines
0, 2, 4, 6) median **46,693,300 ns**, so the median ratio is 1.105×. The individual trials overlap, though:
stride 8 = 41.7, 42.2, 42.3, 48.2, 49.7 ms and stride 16 = 42.2, 43.2, 46.7, 47.0, 47.1 ms. The best-case ratio is
**1.013×**. **This run therefore shows no measurable penalty from 64-byte padding**; the remaining spread is trial noise
on a battery-powered laptop. In theory, stride 16 can only remove an interference that stride 8 has, so it should never
be meaningfully slower.

**Mechanism:** Intel's L2 *spatial (adjacent-line) prefetcher* treats memory as 128-byte-aligned pairs of 64-byte lines.
When a core misses on one line of a pair, the prefetcher also requests the buddy line. AMD Zen has an L2 "up/down"
adjacent-line prefetcher with the same effect. With 64-byte padding, threads 0 and 1 own lines 0 and 1, which form
**one 128-byte pair** (the buffer is 128-B aligned). If Core 0 *misses* on line 0, its prefetcher fetches line 1. That
read steals line 1 from Core 1: M → S on Core 1, which then needs another RFO to write, and the RFO invalidates Core 0's
prefetched copy. These are *secondary invalidations*: false sharing at 128-byte granularity even though no two threads
share a 64-byte line. At 128-byte spacing (lines 0, 2, 4, 6) every thread owns a whole pair, so prefetches never touch
a neighbour's line.

**When padding to 64 B still suffers:** prefetchers act on **misses** (and detected streams). Our loop hits the same
line in L1 on every iteration after the first touch, in M state, so no misses occur and the adjacent-line prefetcher
has nothing to react to. That explains the null result. Padding to 64 B starts to suffer when the working set does
miss: the line gets evicted (large per-thread arrays, other data competing for L1/L2), threads stream through
neighbouring memory, or a thread reads around its slot. In such cases 128-byte padding (or
`std::hardware_destructive_interference_size`, which is 128 on many x86 toolchains for exactly this reason) is the safe
choice.

## Question 4 — Compiler register promotion vs real memory stores

**Why a compiler wants to hoist `buffer[slot]`:** in `naive_promotable_accumulate`, `buffer[slot]` has a loop-invariant
address, and no other memory operation in the loop can alias it. LLVM's LICM *scalar promotion* (plus `mem2reg`)
may legally rewrite the loop as `r = load buffer[slot]; loop { r += … }; store r`, because the language memory model
does not require other threads to see intermediate values. The control kernel in our IR shows this happening and more:
the loop body `vector.body` has **0 loads and 0 stores**. It works on `<8 x i64>` vectors in 4 accumulators
(`%vec.phi … %vec.phi4`, i.e. 32 updates per pass, `add <8 x i64> %vec.ind, splat (i64 -4112007255848680800)` for
the Weyl sequence), and the only store is `store i64 %.179.lcssa, ptr %sunkaddr48` **after** the loop (line 155 of
[`llvm_naive_promotable_accumulate.ll`](llvm_naive_promotable_accumulate.ll)). The run confirms the effect: this
"false-sharing" loop at stride 1 took **8,529,800 ns**, the same as the private baseline and 40× faster than the real
stride-1 kernel. A naive benchmark would therefore wrongly conclude that false sharing does not exist.

**How our kernel prevents it:** `worker_strided_accumulate` computes its address from loop-carried data,
`j = slot + (x & mask)`, where `x` is the Weyl state (`x += 0x9E3779B97F4A7C15`) and `mask` is a run-time argument
equal to 0. The compiler cannot prove that `x & mask` is constant, so `buffer[j]` is a *different, unknown address
each iteration* as far as LLVM knows. It cannot keep the value in a register (the next iteration may read a slot this
iteration wrote) or vectorise the loop. The optimised IR of the native function (`llvm_worker_strided_accumulate.ll`)
shows loop `B34` unrolled 8× with exactly this pattern per update:

```llvm
%.149 = add i64 %x.2.081, %const                  ; x += GAMMA
%.153 = and i64 %.149, %arg.mask                  ; x & mask (opaque to the compiler)
%.181 = getelementptr i64, ptr %0, i64 %.153      ; &buffer[slot + (x & mask)]
%.182 = load i64, ptr %.181, align 8              ; real L1 load
%.183 = lshr i64 %.149, 58
%.184 = add nuw nsw i64 %.183, 1
%.185 = add i64 %.184, %.182
store i64 %.185, ptr %.181, align 8               ; real L1 store, every iteration
```

The audit counts **9 `store i64` + 9 `load i64` inside the loop blocks** (8 unrolled + 1 prologue), 0 volatile/atomic
operations, no calls, and no Python-object API symbols in the native function (`pyobject_fallback: false`; Numba's
`cpython.*` boxing wrapper is excluded because it only runs once per call). Every update is a plain
`load`/`store i64` to memory, which is what makes the RFO traffic in Q1–Q2 real. The private kernel is the opposite
case by design: its accumulator lives in registers, and it also vectorises (0 stores in `vector.body`).

## Question 5 — CPython GIL masking vs `nogil=True`

The same update written in Python bytecode (`python_bytecode_accumulate`, 300,000 updates per thread, CPython 3.14.3
with the GIL enabled):

| Layout | T = 1 | T = 2 | T = 4 | Aggregate ns/update at T = 4 |
|---|---:|---:|---:|---:|
| stride 1 (one line) | 85,328,000 ns | 183,843,700 ns | **361,207,200 ns** | 301.01 |
| stride 8 (separate lines) | 137,826,600 ns | 250,841,700 ns | **360,081,500 ns** | 300.07 |

* **Adding threads makes it slower in wall-clock terms:** stride 1 goes 85 → 184 → 361 ms for 1 → 2 → 4 threads,
  i.e. 4.23× the time for 4× the work. The aggregate throughput never improves (284 → 306 → 301 ns per update), and
  the ~6 % extra comes from GIL hand-offs. Only the thread holding the GIL executes bytecode. The others sleep on the
  GIL's condition variable, and the holder is forced to release it every switch interval (5 ms by default), which
  adds an OS wake-up and context switch to every hand-off.
* **Layout no longer matters:** at T = 4, stride 1 (361.2 ms) and stride 8 (360.1 ms) differ by 0.3 %, while the
  same comparison with `nogil=True` gives 8.04×. Two reasons. (1) The threads never store concurrently: the line moves
  between cores at most once per GIL hand-off (≈ every 5 ms), not once per store. (2) Each update costs ~300 ns of
  interpreter work (bytecode dispatch, int-object allocation, NumPy scalar boxing for `buffer[slot] = acc`), which
  hides the ~5 ns coherence penalty measured in Q2 by about 60×. (The single-thread stride 8 figure, 137.8 ms vs
  85.3 ms, is run-to-run noise of a one-shot measurement on battery power. With one thread the layout cannot matter.)
* **GIL lock contention vs cache-line contention:** GIL contention is a *software mutex* that serialises the
  interpreter. Throughput stays flat or drops as threads are added, whatever the data layout, and the cost is
  paid in OS scheduling. Cache-line contention is a *hardware* effect that only appears when two cores really store
  at the same moment. It depends entirely on layout (8.04× at 8 B spacing, none at 64 B) and is paid in coherence
  messages and store-buffer stalls. `@njit(nogil=True)` removes the first problem: the native loop releases the GIL,
  4 workers run truly in parallel (stride 8 at 4 threads takes 42.3 ms ≈ 1 thread's 40.6 ms), and that is what exposes
  the second. Per update, the native stride-1 kernel (1.416 ns aggregate) is still **≈ 212× faster** than the
  bytecode version (301 ns), even while suffering the worst false sharing.
