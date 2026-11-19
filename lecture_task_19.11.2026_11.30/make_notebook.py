"""Build mpi_cuda_stencil_lab.ipynb from distributed_cuda_solver.py and answers_q1_q6.md.

The %%writefile cell is generated from the solver file so the notebook and the .py never drift.
Run:  python make_notebook.py            (build only; refuses to overwrite an executed notebook
                                         unless --force is given)
      python make_notebook.py --execute  (build, then execute every cell with nbclient)
      python make_notebook.py --answers  (refresh only the Q1-Q6 markdown cell, keep outputs)
"""

import json
import sys
from pathlib import Path


def _src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def new_code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _src(text)}


def new_markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def write_nb(nb):
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "mpi_cuda_stencil_lab.ipynb"
SOLVER = HERE / "distributed_cuda_solver.py"
ANSWERS = HERE / "answers_q1_q6.md"
GITHUB_USERNAME = "Stierlitzx"

TITLE = """# Distributed Multi-Rank Stencil & Asynchronous Halo Exchange in MPI + CUDA (Python)

**Ayazgaliyev Aibek · Student ID 230103341 · GitHub `Stierlitzx` · 19 November 2026**

4-rank (2 × 2 non-periodic Cartesian) anisotropic reaction-diffusion solver on a 512 × 512 grid.
Each rank owns a 258 × 258 padded device array, exchanges halos through GPU pack/unpack kernels
and pinned host buffers with `Isend`/`Irecv`, and updates cells with an 18 × 18 shared-memory
tiled stencil on two CUDA streams (deep interior vs. boundary ring).

Run order: **Runtime → Run all** on a T4 GPU runtime (Colab). The launcher cell also works with
MS-MPI `mpiexec` on Windows for local validation."""

CELL1 = """# CELL 1 - ENVIRONMENT VERIFICATION & MPI4PY INSTALLATION
# (portable form of the three ! commands: works on Colab/Linux and on Windows + MS-MPI)
import shutil, subprocess, sys

print(subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
if shutil.which("mpirun"):
    print("\\n".join(subprocess.run(["mpirun", "--version"], capture_output=True, text=True)
                    .stdout.splitlines()[:2]))
elif shutil.which("mpiexec"):
    print([l for l in subprocess.run(["mpiexec", "-help"], capture_output=True, text=True)
           .stdout.splitlines() if "Version" in l][0])
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mpi4py", "numba", "numpy",
                    "numba-cuda[cu12]"], check=True)
import mpi4py, numba, numpy
print(f"python {sys.version.split()[0]} | numpy {numpy.__version__} | numba {numba.__version__} "
      f"| mpi4py {mpi4py.__version__}")"""

CELL2 = f"""# CELL 2 - STUDENT IDENTITY CONFIGURATION
# Replace with your exact GitHub username before running the solver
GITHUB_USERNAME = "{GITHUB_USERNAME}"
assert GITHUB_USERNAME != "REPLACE_WITH_YOUR_GITHUB_USERNAME" and len(GITHUB_USERNAME.strip()) > 0, (
    "You must set GITHUB_USERNAME to your real GitHub handle."
)
print(f"Configured student GitHub identity: {{GITHUB_USERNAME}}")"""

CELL4 = """# CELL 4 - MULTI-RANK MPI + CUDA EXECUTION LAUNCHER
import shutil, subprocess, sys

if shutil.which("mpirun"):                      # Colab / OpenMPI (instructor command)
    launcher = ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "4"]
    python = "python3"
else:                                           # Windows / MS-MPI local validation
    launcher = ["mpiexec", "-n", "4"]
    python = sys.executable
cmd = launcher + [python, "distributed_cuda_solver.py", GITHUB_USERNAME]

proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
with open("solver_stdout.txt", "w", encoding="utf-8") as fh:
    fh.write(proc.stdout)
if proc.returncode != 0:
    print("STDERR:\\n", proc.stderr)
    raise RuntimeError(f"MPI execution failed with exit code {proc.returncode}")"""

CELL5 = """# Manifest verification summary (reads the file produced by rank 0)
import json
m = json.load(open("submission_manifest.json", encoding="utf-8"))
checks = {
    "GITHUB_USERNAME matches": m["github_username"] == GITHUB_USERNAME,
    "mpi_ranks == 4 in 2x2 grid": m["mpi_ranks"] == 4 and m["cart_dims"] == [2, 2],
    "ptx_audit_passed": m["ptx_audit_passed"],
    "parity_max_diff < 1e-12": m["parity_max_diff"] < 1e-12,
}
for k, v in checks.items():
    print(f"{k:30s} {'OK' if v else 'FAIL'}")
print()
for k in ["gpu_name", "gpu_uuid", "instance_seed", "kappa_max", "dt_max", "dt", "parity_max_diff",
          "serialized_ms", "overlapped_ms", "overlap_speedup", "ptx_counts",
          "energy_monotonic_non_increasing", "manifest_sha256"]:
    print(f"{k:32s} {m[k]}")
assert all(checks.values()), "manifest verification failed"
print("\\n----- submission_manifest.json (verbatim) -----")
print(open("submission_manifest.json", encoding="utf-8").read())"""

CELL6 = """# Excerpt of the audited PTX: every shared-memory store/load and the barrier
import re
ptx = open("tiled_stencil_kernel.ptx", encoding="utf-8").read()
for line in ptx.splitlines():
    if re.search(r"\\b(st|ld)\\.shared\\.|\\bbar(rier)?\\.sync\\b", line):
        print(line.strip())"""


def build_cells():
    solver_src = SOLVER.read_text(encoding="utf-8")
    answers = ANSWERS.read_text(encoding="utf-8") if ANSWERS.exists() else \
        "### Analytical & Forensic Engineering Responses\n\n(answers pending)"
    return [
        new_markdown_cell(TITLE),
        new_code_cell(CELL1),
        new_code_cell(CELL2),
        new_code_cell("%%writefile distributed_cuda_solver.py\n" + solver_src),
        new_code_cell(CELL4),
        new_code_cell(CELL5),
        new_code_cell(CELL6),
        new_markdown_cell(answers),
    ]


def build():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": build_cells(),
    }
    write_nb(nb)
    return nb


def execute(_nb):
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(nb, km=km, timeout=1800, allow_errors=False,
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, NB_PATH)
        km.shutdown_kernel(now=True)


def refresh_answers():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    nb["cells"][-1] = new_markdown_cell(ANSWERS.read_text(encoding="utf-8"))
    write_nb(nb)


if __name__ == "__main__":
    if "--answers" in sys.argv:
        refresh_answers()
    else:
        if NB_PATH.exists() and "--force" not in sys.argv and any(
                c.get("outputs") for c in json.loads(NB_PATH.read_text(encoding="utf-8"))["cells"]):
            raise SystemExit(f"{NB_PATH.name} already holds executed Colab outputs; pass --force to rebuild it.")
        notebook = build()
        if "--execute" in sys.argv:
            execute(notebook)
    print(f"wrote {NB_PATH.name}")
