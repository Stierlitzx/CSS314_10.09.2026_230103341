# Cache Line & False Sharing Mechanics in Multi-Core Python

### Practice lab · 19 November 2026 · 14:30

**Ayazgaliyev Aibek · Student ID `230103341` · Git `Stierlitzx`** · Instructor: Sufyan bin Uzayr

Four GIL-free `@njit(nogil=True)` Numba workers run on `threading.Thread`, each pinned to its own physical core. They
hammer `uint64` counters laid out at 8, 32, 64 and 128-byte spacing to expose hardware cache-coherence (RFO) traffic,
alongside a register-private control and a pure-Python GIL baseline.

**Host:** AMD Ryzen 5 7640HS (Zen 4, 6 C / 12 T), 64-byte L1d lines, Windows 11, CPython 3.14.3, NumPy 2.5.3, Numba 0.67.0.
Run on battery (54 %).

## Results at a glance (4 threads × 60,000,000 updates, median of 5 trials)

| Layout | Cache lines touched | Median wall | ns / update / thread | vs `stride 8` |
|---|---|---:|---:|---:|
| A — `stride 1` (8 B) | `[0]` | 339,939,400 ns | 5.666 | **8.043×** |
| B — `stride 4` (32 B) | `[0, 1]` | 169,377,500 ns | 2.823 | **4.008×** |
| C — `stride 8` (64 B) | `[0, 1, 2, 3]` | 42,262,700 ns | 0.704 | 1.000× |
| D — `stride 16` (128 B) | `[0, 2, 4, 6]` | 46,693,300 ns | 0.778 | 1.105× (trials overlap; best-case 1.013×) |
| Task 4 — register-private | 1 store per thread | **8,506,700 ns** | 0.142 | 0.201× |
| Single thread alone | — | 40,602,800 ns | 0.677 | 0.961× |
| Q4 control — naive promotable loop, stride 1 | (promoted + vectorised) | 8,529,800 ns | 0.142 | 0.202× |

| Check | Result |
|---|---|
| `INSTANCE_SEED` | `16200051438975350361` (`0xe0d224f3c32e7659`) |
| Bit-exact parity (all layouts × 5 trials + private + naive) | **OK** — counters `[1949999939, 1950000037, 1950000048, 1950000014]` |
| LLVM IR of `worker_strided_accumulate` | 9 `load i64` + 9 `store i64` inside the loop, 0 calls, no Python-object fallback |
| GIL released (4-thread stride 16 < 2 × single thread) | **true** |
| **`verification_passed`** | **true** |
| Pure-Python threads (Q5), stride 1, T = 1 / 2 / 4 | 85.3 / 183.8 / 361.2 ms → no speed-up, layout-insensitive |

## Files

| File | Purpose |
|---|---|
| [`false_sharing_lab.py`](false_sharing_lab.py) | The completed lab script (Tasks 1–5); `python false_sharing_lab.py [--quick]` |
| [`cache_lab_manifest.json`](cache_lab_manifest.json) | Telemetry manifest of the full run (`verification_passed: true`, sha256 `e865527cc49da583…`) |
| [`ANALYSIS.md`](ANALYSIS.md) | Answers to Q1–Q5 with the measured numbers and IR excerpts |
| [`full_run_2026-11-19.txt`](full_run_2026-11-19.txt) | Raw console output of the full run |
| `llvm_*.ll` | Optimised LLVM IR of the three kernels, written by the Task 5 audit |

## Implementation notes

* **Task 1:** `INSTANCE_SEED` = first 16 hex digits of SHA-256(`"Stierlitzx::" + cpu_sig`), as the starter's TODO
  requires. Line size comes from sysfs/sysctl and, on Windows, from `GetLogicalProcessorInformation`; 64 is the fallback.
  Physical-core groups come from the same Win32 call (or `/sys/.../topology` on Linux). Each worker pins itself to the
  first logical CPU of a different physical core (`SetThreadAffinityMask` / `sched_setaffinity`). Per-thread seeds are
  `splitmix64(INSTANCE_SEED + t)`.
* **Task 2:** each iteration does `x += 0x9E3779B97F4A7C15; j = slot + (x & mask); buffer[j] += (x >> 58) + 1`, with
  `mask = 0` passed at run time. The address depends on loop-carried data the compiler cannot see through, so LLVM keeps
  a real load and store in every iteration (no register promotion). The 1-cycle Weyl step keeps the memory
  read-modify-write on the critical path.
* **Task 3:** a `threading.Barrier` releases all workers at once. `time.perf_counter_ns()` gives wall time and
  per-thread time. Context switches come from `resource.getrusage` where it exists, otherwise from
  `psutil.num_ctx_switches()` (Windows: voluntary only), sampled while all workers are alive.
* **Task 4:** `worker_private_accumulate` does the same arithmetic in a local register with one store at the end.
* **Task 5:** final counters are compared bit-for-bit with the private reference. `inspect_llvm()` output is parsed for
  self-looping basic blocks of the *native* function, and their `load i64`/`store i64` instructions are counted. The
  check also confirms the native function has no Python C-API calls. All three kernels' IR is saved, and the manifest
  is written.
* **Extras beyond the spec:** a single-thread reference, the naive promotable loop as the Q4 control, the pure-Python
  bytecode baseline for Q5, and the battery/power state in the manifest.

## Reproduction

```powershell
py -3.14 false_sharing_lab.py --quick   # ~2 s sanity run, 2 M updates per thread
py -3.14 false_sharing_lab.py           # full suite, writes cache_lab_manifest.json and llvm_*.ll
```

Timings depend on the machine, power state and background load. `INSTANCE_SEED` changes with the Git handle and CPU
signature, and the final counters change with it.

> Submission protocol note: the assignment asks for a repository named `cache-line-false-sharing-Stierlitzx` containing
> `false_sharing_lab.py`, `cache_lab_manifest.json` and `ANALYSIS.md`. This folder holds those three files.
