# Performance Evaluation & Optimization — Lecture, 29.10.2026, 11:30

**Student:** Ayazgaliyev Aibek, **230103341**. Salted parameters: **N = 1088**, **M = 544**, **seed = 14476**, **16,777,216 float64 buffer elements (128 MiB)**. The complete nine-digit ID is used verbatim by the handout's SHA-256 formula.

## Run

Python 3.10+ and NumPy are required, together with GCC or Clang for the small scalar C kernels loaded through standard-library `ctypes`. The hardware profiler implements the Windows API used on this machine. Run from this folder:

```powershell
python -m pip install -r requirements.txt
python run_all.py --student-id 230103341
python verify_outputs.py
```

`run_all.py` compiles the kernels and runs all milestones in order. Paths are resolved relative to the scripts. All result tables, raw timing samples, numerical output arrays, actual timestamps, compiler settings and logs are saved under `results/`. Native build products stay in ignored `build/` and are rebuilt automatically. No external profiling or Python compilation package is required.

## Milestones and artifacts

| Milestone | Implementation | Outputs |
|---|---|---|
| M0 | `src/profile_env.py`, `src/salt.py` | `host_profile.json`, `salted_params.json` |
| M1 | `src/stride_bench.py`, `src/dense_kernel.py` | `stride_results.tsv`, `loop_interchange.json`, `dense_outputs.npz` |
| M2 | `src/false_sharing.py` | `false_sharing_karp_flatt.tsv`, `counter_outputs.json`, `false_sharing_method.json` |
| M3 | `src/scaling_study.py` | `scaling_metrics.tsv`, `roofline_summary.json`, stencil output/reference `.npy` arrays, `stencil_checksums.json` |
| M4 | `verify_outputs.py` | `verification_receipt.sig`, containing an **unsigned local SHA-256 manifest** |
| Execution | `run_all.py` | `run.log`, `run_manifest.json`, `native_build.json` |

## Measurement design

- **Topology:** actual `GetLogicalProcessorInformation` core masks and cache descriptors, CPU registry model/nominal MHz, `Win32_PhysicalMemory` channel/rate/width telemetry, OS/Python versions, and timer resolution. Microarchitecture is a labelled model-name lookup. Cache hit/miss counters are unavailable and explicitly recorded as uncollected.
- **Stride:** actual scalar accumulation over a 128 MiB contiguous array, five warmups and five recorded runs per stride, median nanoseconds per touched element. Useful bandwidth counts only the elements requested, not inferred cache-line traffic.
- **Loop interchange:** full seeded N×N allocations and M×M computation, preserving row pitch N for both i-j-k and i-k-j. Three runs each, alternating timed order, output zeroing outside timing. The C loops disable fast-math, contraction, and vectorization to isolate scalar traversal order. Both outputs are checked against a separately computed NumPy matrix product with absolute tolerance 1e-12.
- **False sharing:** explicitly 64-byte-aligned lock-free shared int64 storage; unpadded offsets 0,8,…,56; padded offsets 0,64,…; each worker performs exactly 5,000,000 increments. Privatization performs the same volatile native loop in private process memory and one locked final write. `volatile` prevents compiler elimination of the update loop. Three real runs per mode/count. `--backend python` selects the handout's literal Python increment loops; the recorded run uses native inner loops to expose coherence rather than bytecode overhead. Affinity uses separate physical cores first, then SMT: P=8 necessarily includes two SMT siblings on a six-core host.
- **Stencil:** five-point average (four additions + one multiplication), disjoint interior row chunks, zero boundaries and immutable shared input. Each sample repeats the same one-step update 100 times, avoiding cross-chunk halo dependencies. Primary wall time includes start-gate/completion-message overhead; separate lifecycle timing includes process startup, imports, setup, and joins. Three samples for P=1,2,4,6, in strong and weak modes. Weak dimensions use ceil(N√P); actual interior-cell work accounts for rounding/boundaries.
- **Roofline:** report both 5/48 FLOP/B without reuse and 5/16 FLOP/B with ideal reuse, plus write-allocation caveats. A measured large-array NumPy copy supplies an empirical bandwidth reference. Firmware channel/rate/width data supply a theoretical memory payload upper bound, explicitly assuming independently labelled DDR5 channels. The Zen 4 compute model assumes two 256-bit FP64 FMA pipelines and nominal frequency; the scalar kernels do not implement that ideal vectorized workload. Actual DRAM byte traffic is not counted, and the classification is explicitly scoped to this theoretical streaming model.

## Formula corrections and interpretation

For the handout's false-sharing workload, K is fixed **per process**, so total work grows with P. Its requested `S=T1/TP` and `e=(P*TP/T1−1)/(P−1)` are recorded verbatim. Even ideal weak scaling (TP=T1) produces **e=1**, not e≈0. The extra `throughput_speedup=P*T1/TP` and `work_normalized_e=(TP/T1−1)/(P−1)` columns provide a work-normalized comparison; neither is silently substituted for a strong-scaling serial fraction. A default synchronized `multiprocessing.Array` would measure global-lock contention, so shared increments instead use `RawArray` with disjoint writers.

The handout's weak formula `P*W1*T1/(WP*TP)` approaches **weak efficiency** when WP=P·W1. The conventional scaled speedup `WP*T1/(W1*TP)` is included separately. Strong scaling fits Amdahl by least squares in normalized time, reports unconstrained and [0,1]-constrained serial fractions, and includes the fit residual. Gustafson is a labelled model projection, not a claim that weak serial time was measured independently.

## Verification and Git provenance

The instructor's `verify_outputs.py` was not included in the supplied materials. The local verifier independently regenerates seeded matrix/stencil references, checks every saved output, exact individual counters, timing medians and scaling formulas, then hashes source and empirical artifacts. Its `.sig` file is a transparent **unsigned manifest**, not an instructor-authenticated cryptographic signature. Real runtime performance is retained even if it differs from the handout's expected trend.

The specified checkpoint names are `M0-ENV`, `M1-LOCALITY`, `M2-SHARING`, `M3-SCALING`, and `M4-FINAL`. Tags and commit times must represent actual checkpoints. Execution timestamps are recorded as captured; the lecture's 11:30 label does not backdate the run.

## Measured bottleneck findings

On the **AMD Ryzen 5 7640HS (Zen 4, 6 physical cores / 12 logical threads)**, measured caches are **32 KiB L1 data/core, 1 MiB L2/core, 16 MiB shared L3**, with **64-byte cache lines**. The final run's stride latency rises from **1.233 ns/element** at stride 1 to **1.460** at stride 8 and **5.185** at stride 16: the pronounced slowdown occurs later than the handout's simplified stride-8 expectation, consistent with a role for prefetching and memory-level parallelism. Interchanging matrix loops reduces the median from **226.149 ms to 88.433 ms (2.56×)**, with identical scalar outputs and maximum NumPy-reference error **4.27e-13**. At eight workers, unpadded shared counters take **48.338 ms**, versus **4.922 ms padded (9.82× faster)** and **5.870 ms privatized**; all three produce exactly **40,000,000** counts. Strong stencil scaling reaches **5.58× at six cores (92.93% efficiency)**; the fitted Amdahl serial fraction is **2.10%**, giving a model-only asymptotic ceiling of **47.57×**, beyond this host's available cores. Weak scaling reaches only **2.15× conventional scaled speedup** and **35.76% weak efficiency** at six cores as the working set grows. Firmware reports two independent-labelled 64-bit DDR5-5600 channels, yielding a **89.6 GB/s theoretical payload bound**, while the copy calibration measures **52.00 GB/s**. The stencil's **0.104–0.313 FLOP/B** is below the theoretical **4.60 FLOP/B** ridge, placing it on the memory-bound side of the streaming Roofline model; actual DRAM saturation is not established by timing alone.

### Shared-counter medians

| Workers | Unpadded (ms) | Padded (ms) | Privatized (ms) | Unpadded handout e(P) | Padded handout e(P) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.345 | 2.457 | 2.437 | — | — |
| 2 | 34.849 | 3.025 | 2.999 | 28.727 | 1.463 |
| 4 | 23.706 | 2.930 | 3.077 | 13.148 | 1.256 |
| 8 | 48.338 | 4.922 | 5.870 | 23.419 | 2.146 |

Contention clearly collapses throughput, but the unpadded metric is **not monotonic** in P. Padded e(P) is near 1 rather than 0 for the mathematical reason above, and increases when SMT is used. Raw scheduling outliers remain in the TSV rather than being discarded; medians use all three samples.

### Strong and weak stencil scaling

Each time covers 100 independent updates. The strong problem remains 1088×1088; weak problems grow according to the specified dimension formula.

| Workers | Strong median (ms) | Strong speedup | Strong efficiency | Weak N | Weak median (ms) | Conventional weak scaled speedup | Weak efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 144.927 | 1.00× | 100.00% | 1088 | 146.338 | 1.00× | 100.00% |
| 2 | 77.278 | 1.88× | 93.77% | 1539 | 196.192 | 1.49× | 74.59% |
| 4 | 37.102 | 3.91× | 97.65% | 2176 | 259.517 | 2.26× | 56.39% |
| 6 | 25.991 | 5.58× | 92.93% | 2666 | 409.172 | 2.15× | 35.76% |

The fitted Gustafson projection is **5.89×** at six workers; it does not predict the larger working set's observed bandwidth/cache effects. The best strong stencil throughput is **22.69 GFLOP/s**, below the theoretical streaming bound of **28.00 GFLOP/s**. It exceeds the **16.25 GFLOP/s** reference obtained from single-process NumPy copy bandwidth, illustrating that copy calibration is not a universal ceiling for a multi-process stencil with cache reuse.

### Numerical verification

The final complete run took **42.98 seconds**, with actual start/end timestamps in `results/run_manifest.json`. Every local verification check passed: eight stride sums; dense outputs against regenerated references at absolute tolerance 1e-12; all 36 exact counter runs; all 24 stencil runs and their saved arrays; Amdahl fitting and scaling formulas. The matrix digest shared by both loop orders is:

```text
7049e0236c702aa0cf42937108ce8d09e9267e4d98726a7ad8483eabd3626ab8
```
