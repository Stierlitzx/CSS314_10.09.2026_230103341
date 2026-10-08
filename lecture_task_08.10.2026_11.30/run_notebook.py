"""Execute the lab on a physical CUDA GPU and retain every notebook output."""

import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    folder = Path(__file__).resolve().parent
    notebook_path = folder / "CUDA_Lab01_230103341.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)

    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=300,
        resources={"metadata": {"path": str(folder)}},
        allow_errors=False,
    )
    try:
        client.execute()
    finally:
        # Retain diagnostic outputs even if a cell fails.
        nbformat.write(notebook, notebook_path)
        manager.shutdown_kernel(now=True)
    print(f"Executed {notebook_path.name}; results are in {folder}")


if __name__ == "__main__":
    main()
