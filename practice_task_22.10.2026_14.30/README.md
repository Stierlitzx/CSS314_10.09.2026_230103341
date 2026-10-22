# CPU parallelism practice — October 22, 2026, 14:30

**Student:** Ayazgaliyev Aibek — **230103341**

**Session:** Practice, **22.10.2026 at 14:30**, separate from the same day's 11:30 lecture. Execution timestamps in the raw logs record the actual local run time, not a backdated class start.

**Local machine:** AMD Ryzen 5 7640HS, **6 physical cores / 12 logical processors**, Windows 11, approximately 16 GB RAM; **JDK 21.0.2**. Java standard library only. No CUDA, Python, third-party library, internal unmap API or native compiler is required.

## Deliverables and reproduction

| Task | Source | Raw console output | Required commit message |
|---|---|---|---|
| 1 — cache traversal | `Task1.java` | `task1_output.txt` | `feat: task 1 spatial cache profiling` |
| 2 — PRNG state | `Task2.java` | `task2_output.txt` | `feat: task 2 prng state isolation` |
| 3 — inclusive scan | `Task3.java` | `task3_output.txt` | `feat: task 3 parallel scan verification` |
| 4 — interprocess transfer | `Task4.java` | `task4_output.txt` | `feat: task 4 memory transfer profile` |
| 5 — oversubscription | `Task5.java` | `task5_output.txt` | `feat: task 5 oversubscription saturation` |

`PracticeSupport.java` is shared by these five sources and is independent of the lecture folder. `PracticeTests.java` supplies fast deterministic tests, including actual child-process IPC tests. `run_benchmarks.ps1` compiles, tests and saves stdout as UTF-8. Generated class files are ignored by the repository.

```powershell
$lab = 'C:\Users\aiblb\PycharmProjects\CSS314_10.09.2026_230103341\practice_task_22.10.2026_14.30'
& (Join-Path $lab 'run_benchmarks.ps1')             # all five, sequentially
& (Join-Path $lab 'run_benchmarks.ps1') -Task 4     # compile/test, then only Task 4
```

The runner replaces the selected raw output files with fresh measurements. Some full batches exceed 30 seconds; use a terminal without a short command timeout. Do not run benchmarks concurrently. For repeatability, use AC power and stop unrelated CPU-heavy work; the scripts do not change affinity or power settings.

Manual compile/test/run:

```powershell
Set-Location 'C:\Users\aiblb\PycharmProjects\CSS314_10.09.2026_230103341\practice_task_22.10.2026_14.30'
javac -Xlint:all PracticeSupport.java Task1.java Task2.java Task3.java Task4.java Task5.java PracticeTests.java
java -ea -Xmx1g PracticeTests
java -Xms1g -Xmx3g Task1
java -Xms1g -Xmx3g Task2
java -Xms1g -Xmx3g Task3
java -Xms1g -Xmx3g Task4
java -Xms1g -Xmx3g Task5
```

Tasks 1 and 5 obtain physical cores from Windows CIM, summed across processor packages. Elsewhere supply a verified count, for example `java -Dphysical.cores=6 -Xms1g -Xmx3g Task1` (change 6 for that machine); no logical/2 assumption is used. Task 1 uses physical cores. Task 2 uses exactly four threads. Task 3 uses all logical cores. Task 5 explicitly defines `Cores` as logical cores for the requested pool-size formula and prints both hardware counts.

## Task 1 — row-major versus strided traversal

The seeded **8,000 x 8,000 `double[][]`** contains 64 million float64 elements, a **512,000,000-byte payload** plus row/object overhead. Java stores each row separately; its row elements are contiguous, but the matrix is not claimed to be one contiguous 2D allocation. Uniform [0,1) values use `SplittableRandom(230103341)`.

Six workers split rows for Mode A (columns in the inner loop), or split columns for Mode B (rows in the inner loop). All 64 million values are read by each mode. Workers return local sums; the caller merges them. Full serial reference and both parallel sums agree within `max(1,abs(reference))*1e-10`, allowing reduction-order rounding differences.

| Mode | Median of five trials |
|---|---:|
| A — row-major | **12.482800 ms** |
| B — column-major / strided | **127.734100 ms** |
| Cache penalty, B/A | **10.2328x** |

Timing includes job creation/submission, traversal, joins and reduction; excludes matrix allocation/generation, hardware reporting, warmup and pool creation. Variant order alternates. The ratio is evidence of traversal locality cost, not a hardware-counter measurement of cache misses or a claim that the ratio applies on every CPU.

## Task 2 — shared PRNG state versus independent state

Each run evaluates **40,000,000 Monte Carlo points using four workers**. Both variants use `SplittableRandom`, avoiding an algorithm change as a confound. This generator is not thread-safe: the shared version protects the two draws for each point with one `synchronized` block. Independent states are never shared and use exactly these seeds:

```text
thread 0: 2301033410
thread 1: 2301033411
thread 2: 2301033412
thread 3: 2301033413
```

| Version | Computed pi | Absolute error from Math.PI | Median of three trials |
|---|---:|---:|---:|
| Shared PRNG | 3.141251800000 | 0.000340853590 | **1227.108700 ms** |
| Independent PRNG | 3.141417000000 | 0.000175653590 | **43.353799 ms** |
| Ratio of medians | | | **28.3045x** |

Each run starts fresh. Shared point pairs follow one deterministic stream despite scheduling; independent streams are different, so equal pi estimates are neither expected nor asserted. The complete hit counts and each trial's pi/error/timing appear in stdout. Initialization of worker-local generators is included in timing, along with submission, sampling and joins/reduction. Shared state/job setup and persistent pool startup are excluded. Variant order alternates after warmup. This is the cost of state isolation/locking in these implementations, not isolated monitor latency or a test of statistical generator quality.

## Task 3 — work-efficient inclusive scan

Input: **2,000,000 integers in [1,10]**, seeded by the student ID. Outputs use `long` to avoid overflow if the workload grows. The serial implementation performs one inclusive pass. The parallel implementation uses 12 workers for:

1. Local inclusive scans and one total per contiguous chunk.
2. A serial exclusive prefix reduction over the 12 chunk totals.
3. A parallel pass adding each chunk's offset.

This is **O(N+p) work**, with two array passes and two dispatch barriers; it is not a logarithmic-step tree scan. Allocations, input generation, sentinel resets, warmup and correctness checks are excluded from both timings. Parallel job creation, both passes, chunk reduction and joins are included.

| Serial median | Parallel median | Speedup | Verification |
|---:|---:|---:|---|
| **1.728101 ms** | **2.963799 ms** | **0.5831x** | **PASS** |

Both final prefixes are **10,993,592**. Every prefix is checked on **every one of nine measured trials**, not just the last element. Verification explicitly throws `AssertionError` even without `-ea`; the main method prints FAIL and exits **1** on a verification or execution failure. The slower parallel result is retained rather than hidden: two passes, memory traffic and scheduling overhead can outweigh the available parallel work.

## Task 4 — CPU-only interprocess transfer

This implements the assignment's **CPU option**, with a genuine child JVM, not same-process thread messaging.

- **Path A:** serialize the actual `double[]` through an OS pipe using `ObjectOutputStream.reset()` plus `writeUnshared`, so repeated sends do not silently transmit only an object-reference handle.
- **Path B:** populate a temporary raw float64 file from the same parent array, then let the child map it once through `FileChannel.map` / `MappedByteBuffer`. Repeated requests send control messages only; the child reads the mapped array directly. A plain `allocateDirect` buffer would not by itself provide interprocess sharing and is not used as a substitute.

Both timed paths make the child consume **every element in a checksum** and acknowledge a sequence number and checksum. The parent validates the checksum every trial. Timings include payload serialization/copying where applicable, control IPC, child consumption and acknowledgment. They are end-to-end transfer-plus-consumption measurements, not pure isolated bus latency. Generation, child startup, backing-file population/force, mapping/full first-touch and warmup are excluded and reported separately.

| N (float64 elements) | Serialized pipe median | Reused mapping median | A/B ratio |
|---:|---:|---:|---:|
| 100,000 | **4.827800 ms** | **0.629700 ms** | **7.667x** |
| 1,000,000 | **28.169800 ms** | **1.159599 ms** | **24.293x** |
| 10,000,000 | **203.610200 ms** | **9.440900 ms** | **21.567x** |

These are medians of five alternating-order trials; all **30 timed checksums passed**. Setup times, individual end-to-end/child checksum times and raw nanosecond arrays are preserved in stdout. Results demonstrate the benefit of **reusing an already populated mapping**, not that creating/mapping new data is free.

The shared path is **zero repeated payload copy**, not zero setup cost or zero memory traffic. It is a reused file-backed mapping (OS page cache), not a claim of a named anonymous shared-memory facility, direct PCIe profiling or GPU execution. Child exit releases the mapping before file deletion on Windows. Parent never maps, avoiding unsupported unmap APIs. Tests mutate and resend the same array identity and update the existing mapped backing file to verify both paths actually observe new contents. Timeout watchdogs terminate stalled child protocols.

## Task 5 — oversubscription curve

The fixed batch has **10,000 independent tasks**, each computing **500 SHA-256 hashes over a full 1,024-byte payload**. Each digest feeds bytes into the next 1 KiB input, preserving a data dependency rather than repeating an optimizable unused result. Digest state and payload are task-private. A deterministic checksum is checked for every full batch, with unchanged input data across pool sizes.

`Cores = 12 logical processors`, so the unique requested pool sizes are **[1, 6, 12, 24, 48, 192, 256]**. These are real platform threads, not virtual threads. Three trials rotate the order of pool sizes. Pools are prestarted and coexist idle between runs; only one pool executes a batch at any time. Pool startup and provider/code warmup are excluded, but 10,000 task submissions, task allocations, all hashing and joins are included.

| Pool size | Median execution (ms) | Speedup vs 1 | Status |
|---:|---:|---:|---|
| 1 | 2730.911100 | 1.0000x | Baseline |
| 6 | 475.595700 | 5.7421x | Not slower |
| 12 | 309.659199 | 8.8191x | Not slower |
| 24 | 291.113200 | 9.3809x | Oversubscribed; not slower |
| 48 | 286.129899 | 9.5443x | Oversubscribed; not slower |
| 192 | 281.686199 | 9.6949x | Oversubscribed; not slower |
| 256 | 279.071900 | 9.7857x | Oversubscribed; fastest sampled median |

**No worsening was observed among these median pool sizes.** Consequently, there is no measured first degradation size to mark in this saved run. This is a valid negative finding: oversubscription does not guarantee a monotonically degrading wall-time curve for every task/runtime/machine. The program marks the first strictly worse adjacent median when one exists, flags all oversubscribed sizes, reports reversals honestly, and handles ties/no worsening. It never substitutes a theoretical core count for an observed threshold.

The high-pool medians are close; three trials are not a significance test. Per-trial fluctuations are retained. JVM SHA acceleration/code generation, CPU frequency, thermal behavior, task granularity and OS load can affect results. No OS context-switch counters were collected, so timings cannot establish a slowdown **caused specifically** by context switching, even on a future run that observes one. Startup and hundreds of idle workers' memory cost are also outside these steady-state timings.

### Independent end-to-end runner validation

The complete `run_benchmarks.ps1` was subsequently run from compilation through all five full-size benchmarks. It passed, and its entire stdout is retained as **`validation_run_output.txt`**, separate from the primary per-task logs above. The original five logs were restored **byte-for-byte** after this rerun; none of the summary tables silently mixes samples from the two executions.

The validation run **did** observe its first median worsening at **24 threads**: **308.089600 ms at 12 -> 330.630300 ms at 24**. The curve then recovered; **192** was fastest at **264.199600 ms**, while **256** worsened to **273.538900 ms**. This does not replace the original no-worsening result: both runs are reported to demonstrate variability, and neither establishes a universal context-switch threshold or specific causation.

## Validation and interpretation limits

- Compile with `javac -Xlint:all`; standard-library Java 21 only.
- `java -ea -Xmx1g PracticeTests`: **3,375 checks**, plus matrix/numerical assertions and real child-process IPC self-tests, passed.
- Matrix reductions checked against a serial full traversal.
- PRNG tests compare hit counts with serial replay of the exact shared and per-worker seed streams, including uneven partitions and repeated invocations.
- Scan tests cover empty/singleton/uneven inputs, more workers than elements, every prefix, and deliberately corrupted intermediate/final results.
- A fault-injected **temporary copy**, leaving the repository source unchanged, exercised the scan main's failure handler: it printed `Verification Status: FAIL` and exited exactly **1**.
- IPC self-tests use real child processes, repeated array mutations, mapped updates, empty arrays and deterministic Windows file cleanup.
- Hash tests verify partition-independent checksums, no mutation of shared input, required pool sizes, ties/no worsening and a first worsening followed by recovery.
- Raw output retains actual timestamps, hardware/runtime details, samples and formulas. `System.nanoTime()` is an elapsed-time source; nanosecond units do not imply nanosecond resolution.
- All **19 primary raw timing groups** had their medians independently recomputed; reported traversal/PRNG/scan/IPC ratios were checked against those nanosecond samples.
- No artificial delay, forced favorable result, performance assertion, cache flush or CPU pinning is inserted. These are educational benchmarks, not statistically calibrated performance certification.

## Primary technical references

- Oracle Java SE 21 API, `SplittableRandom`: instances are not thread-safe; each worker should use a private instance unless access is externally coordinated.
- Oracle Java SE 21 API, `ObjectOutputStream.reset()` and `writeUnshared()`: prevent reuse of prior serialization stream handles in repeated payload sends.
- Oracle Java SE 21 API, `FileChannel.map()` and `MappedByteBuffer`: file-backed mappings, visibility/OS behavior, and mapping lifetime.
- Oracle Java SE 21 API, `MessageDigest`: SHA-256 updates and digest output; no external hashing library.