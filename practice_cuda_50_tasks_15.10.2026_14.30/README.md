# CSS314 — Advanced GPU Computing with CUDA Python (15.10.2026, started 14:30)

**Student:** Ayazgaliyev Aibek, 230103341<br>
**Course:** CSS314 — Parallel Programming<br>
**Assignment:** 50 Comprehensive Practical Tasks<br>
**Execution target:** Google Colab GPU, Numba CUDA and CuPy, Python 3.10+

Implementations and reference tests for six progressive modules: execution geometry,
memory hierarchy, warp intrinsics and atomics, scientific computing, streams and
CuPy interoperability, and deep-learning primitives.

The original PDF was corrupted. The task text supplied subsequently in chat is the
implementation source; its organized contract summary is in
[`cuda-python-lab/TASK_SHEET.md`](cuda-python-lab/TASK_SHEET.md).

## Repository Structure

```text
practice_cuda_50_tasks_15.10.2026_14.30/
├── README.md
├── CUDA_Python_50_Lab_Tasksheet.pdf   # Original damaged attachment, unchanged
└── cuda-python-lab/
    ├── .github/workflows/autograding.yml
    ├── colab_launcher.ipynb          # Unexecuted Colab bootstrap and full-suite runner
    ├── autograder.py                # Reference suite and machine-readable report
    ├── generate_submission_token.py # Trusted-operator HMAC utility, not official grading
    ├── lab_support.py               # Source hashes and environment metadata
    ├── requirements.txt
    ├── pytest.ini
    ├── TASK_SHEET.md
    ├── lab_kernels/
    │   ├── __init__.py
    │   ├── module1_execution_model.py
    │   ├── module2_memory_hierarchy.py
    │   ├── module3_warp_atomics.py
    │   ├── module4_scientific.py
    │   ├── module5_streams_cupy.py
    │   └── module6_deep_learning.py
    ├── tests/
    │   ├── conftest.py
    │   ├── test_harness.py
    │   └── test_module1.py ... test_module6.py
    └── result/                      # Actual validation logs and JSON reports
```

## How to Run

Open `colab_launcher.ipynb` in Colab, enable a GPU runtime, upload or clone your
assigned repository into `/content`, and run the cells. No repository URL is assumed.

Alternatively, from the lab directory in a CUDA-enabled environment:

```powershell
Set-Location 'C:\Users\aiblb\PycharmProjects\CSS314_10.09.2026_230103341\practice_cuda_50_tasks_15.10.2026_14.30\cuda-python-lab'
python -m pip install -r requirements.txt
python autograder.py
python autograder.py --module 2
```

The full run writes `result/autograder_report.json` including individual outcomes,
hardware identity, software versions, and source hashes. A run with hardware skips
is **incomplete** and returns a nonzero exit code. The suite checks functional
reference results; it is not a replacement for the instructor's private tests.

For local debugging without a CUDA toolkit:

```powershell
python -m pip install numpy numba scipy pytest
python autograder.py --simulator --output result/local_simulator_report.json
```

Simulation is explicitly labeled **not GPU certified**. It cannot establish warp
behavior, transfer overlap, managed-memory migration, graph performance, or timing.
No implementation silently replaces a GPU task with a CPU solution.

## Task Coverage

| Module | Tasks | Implementation focus |
|---|---|---|
| 1 — Execution model | 01–08 | Guarded 1D/2D/3D indexing, grid strides, RGB luminance, LERP, masks |
| 2 — Memory hierarchy | 09–17 | Strides, halos, tiled convolution/GEMM, padded transpose, reductions, constant FIR, pinned transfers |
| 3 — Warp operations & atomics | 18–25 | Shuffle/vote, inclusive scans, private histograms, compaction queue, CAS, sparse dot |
| 4 — Scientific computing | 26–34 | Heat/wave stencils, N-body, Gaussian/Sobel, Monte Carlo, K-means, FFT, Mandelbrot |
| 5 — Streams & CuPy | 35–42 | Dual streams, GEMV pipeline, RawKernel/RawModule, event timing, graphs, managed memory, cooperative barrier |
| 6 — Deep learning | 43–50 | Softmax, fused GELU, pooling, batch/layer norm, CSR SpMV, tiled attention, RoPE |

Each task has a named reference test; some tasks have multiple parameterized cases.
Kernel docstrings specify launch geometry, supported tile/filter limits, and input
layout. Most Numba functions are CUDA dispatchers launched with `kernel[grid, block]`.
Multi-pass operations and benchmarking functions are host orchestration functions.

## Hardware & Validation

The local workstation reports an NVIDIA GeForce RTX 4060 Laptop GPU with 8,188 MiB
VRAM and driver 610.88. Python is 3.14.3. The driver detects the GPU, but Numba
reports `cuda.is_available() == False` and NVVM is unavailable; CuPy is not installed.
Consequently, actual GPU compilation/execution and CuPy-only operations must be
verified in Colab or a fully provisioned CUDA environment.

Actual local validation output is preserved in `cuda-python-lab/result/`. No GPU
throughput, official verification token, figures, or 50-task hardware pass is claimed
unless supported by a recorded real-GPU run.

### Recorded local results (15.10.2026)

| Check | Outcome |
|---|---|
| Full functional simulator suite | **116 passed, 35 skipped, 0 failed** in 62.15 s |
| Non-simulator hardware availability run | **9 passed, 142 skipped**, correctly returned exit code 1 (incomplete) |
| Static task API coverage | All 50 required function names and argument lists present |
| Notebook validation | Valid JSON, code cells compile, no fabricated execution outputs |
| Python syntax | `compileall` completed successfully |
| Git whitespace check | `git diff --check` passed |

The 116 passing cases include parameterized edge cases and CPU-only harness tests;
they do **not** mean 116 CUDA tasks. Simulation exercised 36 of the 50 task APIs.
Tasks **17–19, 24, 31, 35–42, and 47** cannot be exercised by the local simulator,
nor can the large task-sheet workloads in `tests/test_spec_workloads.py`.

- [Simulator JSON report](cuda-python-lab/result/local_simulator_report.json)
- [Full simulator console output](cuda-python-lab/result/local_simulator_output.txt)
- [Hardware availability report](cuda-python-lab/result/local_hardware_check.json)
- [Local environment](cuda-python-lab/result/local_environment.json)

The recorded JSON reports contain source hashes, so later source changes can be
distinguished from this validated snapshot. Their `certified` field is **false**.

### Colab GPU feedback and compatibility correction

The user's Tesla T4 run recorded **130 passed, 21 failed, 0 skipped**. Its full
console output is preserved in `cuda-python-lab/result/colab_before_fix_errors.txt`.
Failures were confined to Tasks10,11,13 (shared allocation shape typing) and
Task42 (C++11 rejected by headers requiring C++17).

The corrected version uses explicit literal shared allocation dimensions and
`--std=c++17` for the cooperative kernel. Regression checks protect both fixes.
No tests were disabled or tolerances weakened. The fixed upload archive is
`cuda-python-lab-colab-fixed.zip` next to this README. Original `local_*` reports
describe pre-fix sources.

Post-fix local simulation: **118 passed, 35 skipped, 0 failed in 60.37 s**,
recorded in `cuda-python-lab/result/post_fix_simulator_report.json`. The two
additional passing tests are compiler-compatibility regression guards.

### Successful post-fix Colab GPU run — 15.10.2026

The user reran the corrected archive in the Colab Tesla T4 session and supplied
the console log: **153 passed, 0 failed, 0 skipped, 95 warnings in 13.75 s**.
The autograder reported `passed_gpu_reference_suite`. The log is preserved in
`cuda-python-lab/result/colab_after_fix_output.txt`.

The warnings report low occupancy for small test grids; they are not test
failures. This is a user-supplied real-GPU reference-suite result, not local GPU
execution or official instructor certification. The original Colab-generated
JSON is preserved unchanged in `cuda-python-lab/result/colab_gpu_report.json`.
All **24 source-manifest SHA256 hashes match the current lab files**. Independent
inspection confirms all 153 individual outcomes passed, no collection errors,
and reference-test coverage for **Tasks 1–50**.

The report was recorded at **2026-10-15 11:55:18 UTC** and records a Tesla T4
(15,360 MiB), driver 580.82.07, Python 3.13.15, Numba 0.61.2, and NumPy 2.1.3.
Its `certified: true` field denotes the lab's own GPU reference-suite acceptance,
not an instructor-issued credential or independent hardware attestation.

### Performance requirements and limitations

The task sheet asks for pinned-transfer speedup, graph-launch speedup, low timing
variance, overlap proof, and shared-bank profiling. Those outcomes depend on the
device, driver, workload, and measurement tools; they are not guaranteed by
numerical correctness. Benchmark utilities return observed measurements rather
than fabricated thresholds. The included functional tests do not certify the
instructor's performance ratios or collect Nsight bank-conflict counters.

## Submission & HMAC Boundary

The supplied text does not contain an instructor secret, public verifier, or actual
official grading harness. `generate_submission_token.py` is therefore a **local
trusted-operator utility**, not an official instructor token generator. It reruns
the complete real-GPU suite, rejects skipped/failed/simulated or stale results,
and signs the report, student identity, hardware metadata, and source hashes.

A trusted operator must supply `LAB_HMAC_SECRET` externally (at least 32 bytes):

```powershell
python generate_submission_token.py 230103341 'Ayazgaliyev Aibek'
```

Never commit a signing secret or place it in a shared notebook. A secret accessible
to a student cannot make a student-generated report trustworthy, and signing GPU
metadata is not hardware attestation. Official acceptance still requires the
instructor's verifier or trusted grading service.

## CI Layout

The parent repository's `.github/workflows/cuda-practice-autograding.yml` runs CPU
simulation on hosted runners. The lab's own `.github/workflows/autograding.yml`
is a standalone-repository template; nested workflows are not active workflows in
this parent repository. Real GPU validation is manually opt-in on a trusted,
pre-provisioned self-hosted runner labeled `gpu`. No self-hosted job runs on
untrusted pull-request events, and CI never publishes signing secrets.