"""Explicit hardware gates: skipped GPU tests never count as certification."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def gpu():
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("Numba CUDA unavailable (driver/toolkit), use Colab or --simulator")
    return cuda


@pytest.fixture(scope="session")
def real_gpu():
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        pytest.skip("Requires real CUDA; warp intrinsics/timing are not simulated")
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("Real Numba CUDA unavailable")
    return cuda


@pytest.fixture(scope="session")
def cupy_gpu():
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        pytest.skip("CuPy tasks require actual GPU execution")
    cp = pytest.importorskip("cupy", reason="CuPy not installed")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA initialization failed: {exc}")
    if count < 1:
        pytest.skip("No CuPy CUDA devices")
    return cp