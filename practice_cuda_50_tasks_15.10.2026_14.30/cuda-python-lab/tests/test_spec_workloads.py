"""Task-sheet workloads reserved for real hardware (never oversized simulation)."""
import numpy as np
import pytest

from lab_kernels import module1_execution_model as execution
from lab_kernels import module2_memory_hierarchy as memory
from lab_kernels import module4_scientific as scientific

pytestmark = [pytest.mark.hardware, pytest.mark.slow]


@pytest.mark.parametrize("n", [1, 255, 256, 1024, 1_000_003])
def test_task01_spec_lengths(real_gpu, n):
    rng = np.random.default_rng(101)
    a, b = (rng.normal(size=n).astype(np.float32) for _ in range(2))
    out = real_gpu.device_array_like(a)
    execution.vector_add_1d[(n + 255) // 256, 256](
        real_gpu.to_device(a), real_gpu.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a + b, atol=1e-5)


def test_task02_spec_grid_stride(real_gpu):
    x = np.linspace(-2, 2, 10_000_003, dtype=np.float32)
    out = real_gpu.device_array_like(x)
    execution.vector_scale_grid_stride[64, 256](real_gpu.to_device(x), np.float32(1.75), out)
    np.testing.assert_allclose(out.copy_to_host(), x * np.float32(1.75), atol=1e-5)


def test_task03_spec_rectangular(real_gpu):
    rng = np.random.default_rng(103)
    a, b = (rng.normal(size=(513, 1021)).astype(np.float32) for _ in range(2))
    out = real_gpu.device_array_like(a)
    execution.matrix_add_2d[((1021 + 15) // 16, (513 + 15) // 16), (16, 16)](
        real_gpu.to_device(a), real_gpu.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a + b, atol=1e-5)


def test_task05_spec_volume(real_gpu):
    d, h, w = 33, 47, 65
    out = real_gpu.device_array((d, h, w), dtype=np.int64)
    execution.volume_init_3d[((w + 7) // 8, (h + 7) // 8, (d + 7) // 8), (8, 8, 8)](
        out, d, h, w)
    z, y, x = np.indices((d, h, w))
    np.testing.assert_array_equal(out.copy_to_host(), z * 10000 + y * 100 + x)


@pytest.mark.parametrize("n", [64, 500, 1024])
def test_task12_spec_matmul(real_gpu, n):
    rng = np.random.default_rng(112)
    a, b = (rng.uniform(-1, 1, (n, n)).astype(np.float32) for _ in range(2))
    out = real_gpu.device_array_like(a)
    memory.matmul_tiled_shared[((n + 15) // 16, (n + 15) // 16), (16, 16)](
        real_gpu.to_device(a), real_gpu.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a @ b, rtol=1e-4, atol=1e-3)


def test_task31_spec_monte_carlo(cupy_gpu):
    cupy_gpu.random.seed(131)
    estimate = scientific.monte_carlo_pi_cupy(100_000_000)
    assert abs(estimate - np.pi) < 1e-3