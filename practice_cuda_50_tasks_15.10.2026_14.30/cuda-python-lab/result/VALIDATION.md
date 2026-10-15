# Validation History — 15.10.2026

## Environment

- Windows 11, Python 3.14.3
- NumPy 2.5.3, Numba 0.67.0, SciPy 1.18.1, PyTest 9.1.1
- NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB, driver 610.88
- `cuda.is_available() == False`; NVVM unavailable; CuPy not installed

## Full Simulator Run

Command: `python autograder.py --simulator --output result/local_simulator_report.json`

**116 passed, 35 skipped, 0 failed in 62.15 seconds.**

Status: `simulated_not_certified`; PyTest exit code 0; `certified: false`.
Tests use actual Numba CUDA simulator execution of supported kernels, not an
alternate CPU implementation. Independent references use NumPy and SciPy.

The suite includes task signatures and harness checks, parameterized edge cases,
and large real-GPU-only workload checks. There are 50 task APIs, not 151 tasks.
Simulation exercised 36 APIs. Tasks17,18,19,24,31,35–42,47 cannot be validated by
this simulator; hardware coverage is recorded in the Colab sections below. The 35 skipped test cases
also include large hardware-only workloads.

## Real-Hardware Availability Check

Command: `python autograder.py --output result/local_hardware_check.json`

**9 passed, 142 skipped, 0 failed in 3.55 seconds.**

Only CPU harness/static-contract tests passed. Hardware tasks skipped because the
CUDA compiler runtime is unavailable and CuPy is absent. The autograder correctly
returned **exit code 1**, status `incomplete_or_failed`, `certified: false`.

## Additional Checks

- All 50 function signatures match `TASK_SHEET.md` argument names.
- Every task has a named reference test.
- Notebook is valid JSON; its Python cells compile and are unexecuted.
- `python -m compileall -q` succeeded for the lab.
- `git diff --check` passed.
- The copied original attachment still matches its original SHA256 hash.

## User-Supplied Colab GPU Run (Before Compatibility Fix)

The supplied complete console log is preserved as `colab_before_fix_errors.txt`.
It records **130 passed, 21 failed, 0 skipped in 22.81 seconds** on the user's
Colab session. This is a user-run GPU result, not local hardware execution.

The 21 failures affect four task functions:

- Task10 (5 cases), Task11 (5 cases), Task13 (8 cases): CUDA typing rejects
  arithmetic expressions in static shared allocation shapes as nonliteral integers.
- Task42 (3 cases): cooperative-group headers reject the explicit C++11 option
  because the installed libcu++ requires C++17.

Compatibility corrections use explicit literal shared array dimensions throughout
Module2 and request `--std=c++17` for the cooperative RawKernel. No functional
assertions or tolerances were removed or weakened. Two CPU regression guards
check the shared-allocation literals and cooperative compiler options.

The original `local_*` JSON reports above remain historical snapshots of the
pre-fix sources. The new snapshot is recorded in `post_fix_simulator_report.json`;
source manifests intentionally differ between the two versions.

Post-fix local full simulator run: **118 passed, 35 skipped, 0 failed in 60.37
seconds**, status `simulated_not_certified`. This includes the two added
compiler-compatibility regression guards. Python syntax and Git whitespace
checks also passed. This does not constitute a post-fix GPU execution result.

## Successful User-Supplied Post-Fix Colab GPU Run — 15.10.2026

The user uploaded the corrected archive and reran the autograder in the Colab
Tesla T4 session. The supplied console log is preserved unchanged as
`colab_after_fix_output.txt`.

- **153 passed, 0 failed, 0 skipped, 95 warnings in 13.75 seconds.**
- Reported status: `passed_gpu_reference_suite`.
- Report path in Colab: `/content/cuda-python-lab/result/autograder_report.json`.
- The 95 `NumbaPerformanceWarning` messages flag small-grid GPU under-utilization;
  they are not test failures or measured performance guarantees.

This establishes a successful user-reported full GPU reference-suite run after
the compatibility fixes, not local GPU execution or instructor certification.
The actual GPU JSON report is now preserved byte-for-byte as
`colab_gpu_report.json`. Local inspection confirms:

- All **24 SHA256 source-manifest entries match the current lab files**, including
  the six kernel modules, reference tests, harness, and launcher.
- All **153 individual outcomes are passed**, with no collection errors.
- `passed_task_numbers` and named passing reference tests both cover Tasks 1–50.
- `mode: real_gpu`, `module: null`, `pytest_exit_code: 0`, and
  `status: passed_gpu_reference_suite`.
- Recorded timestamp: **2026-10-15 11:55:18.088590 UTC**.
- Recorded environment: Tesla T4, 15,360 MiB, driver 580.82.07; Linux,
  Python 3.13.15, Numba 0.61.2, NumPy 2.1.3.
- Original report SHA256:
  `3a6ab2f9bbf30f852866e325e732cd58e5df62667b8177a1cd8883d9b7eca05e`.

The report's `certified: true` is the lab's own reference-suite acceptance flag,
not official instructor certification or hardware attestation. The environment
values above are report metadata, not independently observed local hardware.
Local simulator reports remain unchanged and explicitly simulation-only.

The tests do not measure Nsight bank-conflict counters or prove prescribed
performance ratios/stream overlap. Those remain external profiling checks.
No official HMAC credential has been generated: the instructor's trusted verifier
and signing infrastructure were not provided.

Reports carry SHA256 source manifests for the tested snapshot. Changes after a run
must be validated again. GPU identity in a JSON report is metadata, not hardware
attestation.