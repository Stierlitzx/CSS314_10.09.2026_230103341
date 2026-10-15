"""Reproducible source manifests and best-effort execution environment metadata."""
import hashlib
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def source_manifest():
    paths = list((ROOT / "lab_kernels").glob("*.py"))
    paths += list((ROOT / "tests").glob("*.py"))
    paths += [ROOT / name for name in (
        "autograder.py", "generate_submission_token.py", "lab_support.py",
        "requirements.txt", "pytest.ini", "colab_launcher.ipynb",
    )]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def environment():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15,
            check=True,
        )
        info["nvidia_smi"] = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        info["nvidia_smi_error"] = str(exc)
    try:
        import numba
        import numpy
        info.update(numba=numba.__version__, numpy=numpy.__version__)
    except ImportError as exc:
        info["import_error"] = str(exc)
    return info