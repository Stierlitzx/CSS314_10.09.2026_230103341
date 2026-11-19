"""Build hybrid_mpi_cuda_cache_lab.ipynb from hybrid_mpi_cuda_cache.py (+ answers_q1_q6.md).

  python make_notebook.py            build (refuses to overwrite an executed notebook unless --force)
  python make_notebook.py --answers  replace only the Q1-Q6 Markdown cell, keeping all outputs
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "hybrid_mpi_cuda_cache_lab.ipynb"
SCRIPT = HERE / "hybrid_mpi_cuda_cache.py"
ANSWERS = HERE / "answers_q1_q6.md"
GITHUB_USERNAME = "Stierlitzx"


def _src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _src(text)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


TITLE = """# Hybrid MPI-CUDA Distributed Stencil & Cross-Tier Cache-Line Coherency (Python)

**Ayazgaliyev Aibek · Student ID 230103341 · GitHub `Stierlitzx` · 19 November 2026 (practice, 14:30)**

Four MPI ranks form a non-periodic 2 × 2 Cartesian grid over a 512 × 512 domain. Each rank owns a 258 × 258 padded
GPU array and exchanges halos via GPU pack/unpack kernels, pinned buffers and `Isend`/`Irecv`. Cells are updated by an
18 × 18 shared-memory tiled stencil on two CUDA streams. The same 4 ranks then share one 64-byte-aligned
`MPI.Win.Allocate_shared` window and hammer `uint64` counters at 8 / 64 / 128-byte spacing from a
`@njit(nogil=True)` kernel to measure host MESI false sharing.

Run with **Runtime → Run all** on a **T4 GPU** runtime."""

CELL1 = """# CELL 1 - HARDWARE & MPI ENVIRONMENT SETUP
!nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total --format=csv
!lscpu | grep -E "Model name|CPU\\(s\\)|cache" || true
!pip install -q mpi4py numba numpy "numba-cuda[cu12]"
!mpirun --version | head -n 1"""

CELL2 = f"""# CELL 2 - SOLO STUDENT IDENTITY CONFIGURATION
GITHUB_USERNAME = "{GITHUB_USERNAME}"
assert GITHUB_USERNAME != "REPLACE_WITH_YOUR_GITHUB_USERNAME" and len(GITHUB_USERNAME.strip()) > 0, (
    "Set GITHUB_USERNAME to your real GitHub handle before running."
)
print(f"Verified solo student handle: {{GITHUB_USERNAME}}")"""

CELL4 = """# CELL 4 - MULTI-RANK HYBRID EXECUTION CELL
import subprocess
cmd = [
    "mpirun",
    "--allow-run-as-root",
    "--oversubscribe",
    "-np", "4",
    "python3",
    "hybrid_mpi_cuda_cache.py",
    GITHUB_USERNAME,
]
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
with open("hybrid_stdout.txt", "w", encoding="utf-8") as fh:
    fh.write(proc.stdout)
if proc.returncode != 0:
    print("STDERR:\\n", proc.stderr)
    raise RuntimeError(f"Hybrid MPI-CUDA execution failed with code {proc.returncode}")"""

CELL5 = """# Manifest verification summary + verbatim manifest
import json
m = json.load(open("hybrid_lab_manifest.json", encoding="utf-8"))
checks = dict(m["verification"])
checks["github_username matches"] = m["github_username"] == GITHUB_USERNAME
for k, v in checks.items():
    print(f"{k:28s} {'OK' if v else 'FAIL'}")
print()
for k in ["gpu_name", "gpu_uuid", "cpu_model", "instance_seed", "kappa_max", "dt_max", "dt", "parity_max_diff",
          "gpu_serialized_ms", "gpu_overlapped_ms", "gpu_overlap_speedup", "ptx_counts",
          "shm_false_sharing_slowdown_stride1_vs_stride8", "shm_ratio_stride16_vs_stride8",
          "rank_cpu_affinity", "verification_passed"]:
    print(f"{k:46s} {m[k]}")
assert m["verification_passed"] and checks["github_username matches"], "verification failed"
print("\\n----- hybrid_lab_manifest.json (verbatim) -----")
print(open("hybrid_lab_manifest.json", encoding="utf-8").read())"""

CELL6 = """# Audit excerpts: PTX shared-memory/barrier instructions and the LLVM IR loop body of the host kernel
import re
for line in open("tiled_stencil_kernel.ptx", encoding="utf-8"):
    if re.search(r"\\b(st|ld)\\.shared\\.|\\bbar(rier)?\\.sync\\b", line):
        print(line.strip())
print("-" * 60)
ir = open("host_shm_accumulate.ll", encoding="utf-8").read()
body = next(f for f in ir.split("\\ndefine ")[1:] if "host_shm_accumulate" in f.split("(", 1)[0]
            and "cpython" not in f.split("(", 1)[0] and "cfunc" not in f.split("(", 1)[0])
start = body.find("\\nB34:")
print(body[start:start + 1400] if start >= 0 else body[:1400])"""


def build():
    answers = ANSWERS.read_text(encoding="utf-8") if ANSWERS.exists() else \
        "### Analytical & Forensic Engineering Responses\n\n(answers pending)"
    cells = [md(TITLE), code(CELL1), code(CELL2),
             code("%%writefile hybrid_mpi_cuda_cache.py\n" + SCRIPT.read_text(encoding="utf-8")),
             code(CELL4), code(CELL5), code(CELL6), md(answers)]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"}},
          "cells": cells}
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def refresh_answers():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    nb["cells"][-1] = md(ANSWERS.read_text(encoding="utf-8"))
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--answers" in sys.argv:
        refresh_answers()
    else:
        if NB_PATH.exists() and "--force" not in sys.argv and any(
                c.get("outputs") for c in json.loads(NB_PATH.read_text(encoding="utf-8"))["cells"]):
            raise SystemExit(f"{NB_PATH.name} already holds executed outputs; pass --force to rebuild.")
        build()
    print(f"wrote {NB_PATH.name}")
