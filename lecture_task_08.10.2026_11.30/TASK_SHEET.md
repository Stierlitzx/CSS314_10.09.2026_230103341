# CUDA Lecture Task Sheet

Student: Aibek Ayazgaliyev | Student ID: 230103341
Session: Lecture | Date / start time: 08.10.2026 11:30
User-approved local RTX4060 execution instead of the assigned Colab T4; no Colab completion claimed.
Measured GPU: index 0, UUID GPU-0c13a78b-5a3e-7434-f187-59658df572e9

## 1. Hardware Telemetry

| Field | Measured value |
|---|---|
| NVIDIA driver | 610.88 |
| Driver-supported CUDA | 13.3 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8188 MiB |
| Compute capability | 8.9 |
| Maximum threads per block | 1024 |

## 2. Vector Doubling

| N | CPU ms | GPU total ms | GPU kernel only ms | Winner |
|---:|---:|---:|---:|---|
| 100 | 0.045700 | 2.138000 | 0.134300 | CPU |
| 1000 | 0.245000 | 0.974000 | 0.165500 | CPU |
| 10000 | 3.668100 | 1.165500 | 0.190700 | GPU |
| 100000 | 30.719000 | 1.004000 | 0.144300 | GPU |
| 1000000 | 295.895400 | 3.320900 | 0.375000 | GPU |
| 5000000 | 1513.191400 | 11.936000 | 0.504400 | GPU |
| 10000000 | 2868.926800 | 19.743300 | 4.137800 | GPU |

Q1. The first tested size at which GPU total beats CPU is N* = 10,000.

Q2. At N=100, GPU total is 2.138000 ms, versus kernel 0.134300 ms. The difference is approximately 93.72% of total time. Allocation, PCIe transfers, launch, and synchronization can dominate tiny workloads; these separately sampled intervals do not isolate PCIe time or guarantee a 95% overhead fraction.

CPU measured at all sizes rather than optionally skipped. GPU times use synchronized perf_counter; kernel-only uses preloaded arrays and includes launch/synchronization overhead.

## 3. Boundaries and Thread Geometry

N=1000: 4 blocks x 256 threads = 1024 threads; 24 writers have no valid element.

Exact defective-kernel console observation (isolated process on the same GPU):
```text
Child exit code: 0
Defective experiment GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Launching defective kernel with 1024 threads on 1000-element array...
C:\Users\aiblb\AppData\Local\Temp\opencode\cuda-230103341\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 4 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
Execution finished (or did it crash?). Checking host copy...
Elements processed: 1000. Last element: 999.0
```

A successful host copy does not prove the out-of-bounds writes were safe.

N=37: blocks=5, launched=40, active=37, idle=3.

| Global ID | Block ID | Thread ID | Status |
|---:|---:|---:|---|
| 32 | 4 | 0 | ACTIVE |
| 33 | 4 | 1 | ACTIVE |
| 34 | 4 | 2 | ACTIVE |
| 35 | 4 | 3 | ACTIVE |
| 36 | 4 | 4 | ACTIVE |
| 37 | 4 | 5 | IDLE (GUARDED) |
| 38 | 4 | 6 | IDLE (GUARDED) |
| 39 | 4 | 7 | IDLE (GUARDED) |

## 4. Parallel Reduction

| Trial | Target sum | Naive output | Lost updates | Kernel ms |
|---:|---:|---:|---:|---:|
| 1 | 50000.0 | 2.0 | 49998.0 | 0.120800 |
| 2 | 50000.0 | 2.0 | 49998.0 | 0.090200 |
| 3 | 50000.0 | 2.0 | 49998.0 | 0.078000 |
| 4 | 50000.0 | 2.0 | 49998.0 | 0.077800 |
| 5 | 50000.0 | 2.0 | 49998.0 | 0.075900 |

Atomic sum: 50000.0. Exactly 50000.0: True.
Median of five trials: naive=0.078000 ms; atomic=0.154700 ms.
Atomic / naive time ratio: 1.983333207403305.

Atomic addition prevents lost read-modify-write updates. All threads contend for the same scalar address, so updates are serialized rather than proceeding independently. The actual penalty depends on hardware, compiler optimizations, and timing noise; an incorrect naive result is not an equivalent correct algorithm.

## 5. 2D Geometry

Blocks X=64, blocks Y=64, total blocks=4096.
Verified center (512,512)=1.000000000; outer corner (0,0)=0.000000000.
The full GPU canvas passed the NumPy reference comparison.

## 6. Final Token and Submission

**TOKEN: DF248880DE203CB7**

- [ ] Executed notebook saved as CUDA_Lab01_230103341.ipynb with outputs retained.
- [ ] All cells executed sequentially with visible outputs retained.
- [ ] Executed .ipynb uploaded to the course portal.
- [ ] This empirical task sheet reviewed and submitted to the instructor.
