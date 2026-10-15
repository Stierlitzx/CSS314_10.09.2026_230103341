"""NumPy-reference tests for tasks 01–08 (CUDA or explicitly enabled CUDASIM)."""

import numpy as np
import pytest

from lab_kernels.module1_execution_model import (
    matrix_add_2d,
    matrix_transpose_naive,
    rgb_to_grayscale,
    vector_add_1d,
    vector_lerp,
    vector_scale_grid_stride,
    vector_threshold,
    volume_init_3d,
)


@pytest.mark.parametrize("n", [0, 1, 31, 259])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_task01_vector_add(gpu, n, dtype):
    rng = np.random.default_rng(1)
    a, b = (rng.standard_normal(n).astype(dtype) for _ in range(2))
    out = gpu.to_device(np.full(n, np.nan, dtype=dtype))
    vector_add_1d[max(1, (n + 255) // 256), 256](gpu.to_device(a), gpu.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a + b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("n,alpha", [(0, 2), (1, 0), (17, -2.5), (1003, 3.25)])
def test_task02_grid_stride(gpu, n, alpha):
    x = np.linspace(-3, 5, n, dtype=np.float32)
    out = gpu.to_device(np.full(n, np.nan, dtype=np.float32))
    # Deliberately insufficient for one-thread-per-element indexing.
    vector_scale_grid_stride[2, 16](gpu.to_device(x), np.float32(alpha), out)
    np.testing.assert_allclose(out.copy_to_host(), x * alpha, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 7), (1, 1), (3, 19), (17, 5)])
def test_task03_matrix_add(gpu, shape):
    rng = np.random.default_rng(3)
    a, b = (rng.standard_normal(shape).astype(np.float32) for _ in range(2))
    out = gpu.to_device(np.full(shape, np.nan, dtype=np.float32))
    grid = (max(1, (shape[1] + 15) // 16), max(1, (shape[0] + 15) // 16))
    matrix_add_2d[grid, (16, 16)](gpu.to_device(a), gpu.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a + b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shape", [(0, 7), (1, 1), (3, 19), (17, 5)])
def test_task04_naive_transpose(gpu, shape):
    a = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    out = gpu.to_device(np.full(shape[::-1], np.nan))
    grid = (max(1, (shape[1] + 7) // 8), max(1, (shape[0] + 3) // 4))
    matrix_transpose_naive[grid, (8, 4)](gpu.to_device(a), out)
    np.testing.assert_array_equal(out.copy_to_host(), a.T)


@pytest.mark.parametrize("shape", [(0, 2, 3), (1, 1, 1), (3, 5, 9)])
def test_task05_volume_init(gpu, shape):
    d, h, w = shape
    out = gpu.to_device(np.full(shape, -1, dtype=np.int64))
    grid = (max(1, (w + 7) // 8), max(1, (h + 7) // 8), max(1, (d + 7) // 8))
    volume_init_3d[grid, (8, 8, 8)](out, d, h, w)
    z, y, x = np.indices(shape)
    np.testing.assert_array_equal(out.copy_to_host(), z * 10000 + y * 100 + x)


@pytest.mark.parametrize("shape", [(0, 4), (1, 1), (5, 11)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_task06_rgb_to_grayscale(gpu, shape, dtype):
    rgb = np.random.default_rng(6).integers(0, 256, (*shape, 3)).astype(dtype)
    out = gpu.to_device(np.full(shape, np.nan, dtype=np.float32))
    grid = (max(1, (shape[1] + 3) // 4), max(1, (shape[0] + 3) // 4))
    rgb_to_grayscale[grid, (4, 4)](gpu.to_device(rgb), out)
    expected = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    np.testing.assert_allclose(out.copy_to_host(), expected, rtol=1e-6, atol=2e-5)


@pytest.mark.parametrize("n", [0, 1, 35, 101])
def test_task07_vector_lerp(gpu, n):
    rng = np.random.default_rng(7)
    a, b = (rng.standard_normal(n) for _ in range(2))
    t = np.resize(np.array([0.0, 1.0, 0.3, -0.5, 1.5]), n)
    out = gpu.to_device(np.full(n, np.nan))
    vector_lerp[max(1, (n + 31) // 32), 32](gpu.to_device(a), gpu.to_device(b), gpu.to_device(t), out)
    np.testing.assert_allclose(out.copy_to_host(), (1 - t) * a + t * b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
@pytest.mark.parametrize("n", [0, 1, 37])
def test_task08_vector_threshold(gpu, dtype, n):
    x = np.resize(np.array([-np.inf, -1, 0.5, 0.5001, np.inf, np.nan]), n)
    out = gpu.to_device(np.ones(n, dtype=dtype))
    vector_threshold[1, 8](gpu.to_device(x), 0.5, out)
    np.testing.assert_array_equal(out.copy_to_host(), (x >= 0.5).astype(dtype))