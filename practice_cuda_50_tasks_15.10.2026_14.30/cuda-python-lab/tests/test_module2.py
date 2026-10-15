"""Tasks 09–17: independent references, boundary tiles and deterministic ties."""

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from lab_kernels.module2_memory_hierarchy import (
    WEIGHTS,
    benchmark_pinned_transfer,
    benchmark_strided_access,
    constant_memory_filter,
    conv_1d_shared_tiled,
    matmul_tiled_shared,
    reduction_argmax_shared,
    reduction_sum_shared,
    stencil_1d_shared,
    transpose_bank_conflict_free,
)


def _same_signal_length(x, weights):
    """Explicit crop avoids np.convolve's max(N,K) rule for short signals."""
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    full = np.convolve(x, weights, mode="full")
    start = (len(weights) - 1) // 2
    return full[start:start + x.size]


@pytest.mark.parametrize("n,stride", [(0, 1), (1, 7), (37, 1), (37, 2), (37, 4), (37, 8), (37, 16), (37, 32)])
def test_task09_strided_access(gpu, n, stride):
    x = np.arange(max(0, (n - 1) * stride + 1), dtype=np.float32) + 0.25
    out = gpu.to_device(np.full(n, np.nan, dtype=np.float32))
    benchmark_strided_access[max(1, (n + 31) // 32), 32](gpu.to_device(x), out, stride)
    np.testing.assert_array_equal(out.copy_to_host(), x[:n * stride:stride])


@pytest.mark.parametrize("n,radius,threads", [(0, 3, 16), (1, 32, 8), (19, 0, 16), (35, 3, 16), (273, 32, 256)])
def test_task10_shared_stencil(gpu, n, radius, threads):
    x = np.random.default_rng(10).standard_normal(n)
    out = gpu.to_device(np.full(n, np.nan))
    stencil_1d_shared[max(1, (n + threads - 1) // threads), threads](gpu.to_device(x), out, radius)
    expected = uniform_filter1d(x, size=2 * radius + 1, mode="nearest") if n else x.copy()
    np.testing.assert_allclose(out.copy_to_host(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n,k,threads", [(0, 3, 16), (1, 65, 8), (37, 1, 16), (71, 4, 16), (79, 65, 32)])
def test_task11_shared_convolution(gpu, n, k, threads):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n).astype(np.float32)
    weights = (np.arange(1, k + 1) / (k * k)).astype(np.float32)
    out = gpu.to_device(np.full(n, np.nan, dtype=np.float32))
    conv_1d_shared_tiled[max(1, (n + threads - 1) // threads), threads](
        gpu.to_device(x), gpu.to_device(weights), out)
    np.testing.assert_allclose(out.copy_to_host(), _same_signal_length(x, weights), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (3, 0, 5), (17, 19, 9), (7, 5, 33)])
def test_task12_tiled_matmul(gpu, m, k, n):
    rng = np.random.default_rng(12)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    out = gpu.to_device(np.full((m, n), np.nan, dtype=np.float32))
    matmul_tiled_shared[((n + 15) // 16, (m + 15) // 16), (16, 16)](
        gpu.to_device(a), gpu.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a @ b, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("shape", [(0, 5), (1, 1), (17, 35), (33, 7)])
@pytest.mark.parametrize("block_y", [8, 32])
def test_task13_padded_transpose(gpu, shape, block_y):
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    out = gpu.to_device(np.full(shape[::-1], np.nan, dtype=np.float32))
    grid = (max(1, (shape[1] + 31) // 32), max(1, (shape[0] + 31) // 32))
    transpose_bank_conflict_free[grid, (32, block_y)](gpu.to_device(a), out)
    np.testing.assert_array_equal(out.copy_to_host(), a.T)


@pytest.mark.parametrize("n", [0, 1, 255, 256, 513])
def test_task14_shared_sum(gpu, n):
    x = np.random.default_rng(14).standard_normal(n)
    blocks = max(1, (n + 255) // 256)
    out = gpu.to_device(np.full(blocks, np.nan))
    reduction_sum_shared[blocks, 256](gpu.to_device(x), out)
    expected = np.array([x[i * 256:(i + 1) * 256].sum() for i in range(blocks)])
    np.testing.assert_allclose(out.copy_to_host(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.copy_to_host().sum(), x.sum(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n,all_negative_inf", [(0, False), (1, False), (257, False), (513, False), (257, True)])
def test_task15_shared_argmax(gpu, n, all_negative_inf):
    x = -np.arange(n, dtype=np.float64) - 1000
    if all_negative_inf:
        x[:] = -np.inf
    elif n > 256:
        # Equal maxima in each block and at nonadjacent tree leaves.
        x[1] = x[127] = x[256] = 20
    blocks = max(1, (n + 255) // 256)
    values = gpu.to_device(np.full(blocks, np.nan))
    indices = gpu.to_device(np.full(blocks, -99, dtype=np.int64))
    reduction_argmax_shared[blocks, 256](gpu.to_device(x), values, indices)
    expected_values, expected_indices = [], []
    for block in range(blocks):
        part = x[block * 256:(block + 1) * 256]
        idx = int(np.argmax(part)) if part.size else -1
        expected_values.append(part[idx] if part.size else -np.inf)
        expected_indices.append(block * 256 + idx if part.size else -1)
    np.testing.assert_array_equal(values.copy_to_host(), expected_values)
    np.testing.assert_array_equal(indices.copy_to_host(), expected_indices)


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 137])
def test_task16_constant_filter(gpu, n):
    assert WEIGHTS.shape == (64,)
    np.testing.assert_allclose(WEIGHTS.sum(), 1.0, atol=1e-7)
    np.testing.assert_array_equal(WEIGHTS, np.arange(1, 65, dtype=np.float32) / np.float32(2080))
    x = np.random.default_rng(16).standard_normal(n).astype(np.float32)
    out = gpu.to_device(np.full(n, np.nan, dtype=np.float32))
    constant_memory_filter[max(1, (n + 31) // 32), 32](gpu.to_device(x), out)
    np.testing.assert_allclose(out.copy_to_host(), _same_signal_length(x, WEIGHTS), rtol=2e-5, atol=1e-6)


@pytest.mark.hardware
@pytest.mark.parametrize("size_bytes", [4096, 1024 * 1024])
def test_task17_pinned_transfer(real_gpu, size_bytes):
    result = benchmark_pinned_transfer(size_bytes)
    assert result["size_bytes"] == size_bytes
    assert result["repetitions"] == 5
    for memory in ("pageable", "pinned"):
        for direction in ("h2d", "d2h"):
            key = memory + "_" + direction
            assert np.isfinite(result[key + "_ms"]) and result[key + "_ms"] > 0
            assert np.isfinite(result[key + "_wall_ms"]) and result[key + "_wall_ms"] > 0
            np.testing.assert_allclose(result[key + "_gbps"], size_bytes / (result[key + "_ms"] * 1e6))
    # Do not assert pinned is faster: hardware and driver overhead vary.
    for direction in ("h2d", "d2h"):
        np.testing.assert_allclose(result[direction + "_speedup"],
                                   result["pageable_" + direction + "_ms"] / result["pinned_" + direction + "_ms"])