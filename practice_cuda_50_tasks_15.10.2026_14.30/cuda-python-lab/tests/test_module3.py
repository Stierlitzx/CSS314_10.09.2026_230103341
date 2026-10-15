"""NumPy reference checks for tasks18--25; no CPU substitutes for kernels."""

import numpy as np

from lab_kernels import module3_warp_atomics as m


def test_task18(real_gpu):
    cuda = real_gpu
    for n, threads in ((1, 32), (31, 64), (32, 64), (33, 64), (97, 64), (259, 256)):
        x = (np.arange(n, dtype=np.float32) - 17) / 7
        blocks = (n + threads - 1) // threads
        out = cuda.device_array(blocks, dtype=np.float32)
        m.warp_reduce_sum_shfl[blocks, threads](cuda.to_device(x), out)
        padded = np.pad(x, (0, blocks * threads - n)).reshape(-1, threads)
        np.testing.assert_allclose(out.copy_to_host(), padded.sum(axis=1), rtol=2e-6, atol=2e-5)


def test_task19(real_gpu):
    cuda = real_gpu
    # Bit31 set, zeros and negative values false; second warp only one lane.
    x = np.full(33, -1, dtype=np.int32)
    x[[0, 7, 31, 32]] = 2
    x[1] = 0
    out = cuda.device_array(2, dtype=np.uint32)
    m.warp_vote_analyzer[1, 64](cuda.to_device(x), out)
    reference = np.array([sum(1 << j for j in (0, 7, 31)), 1], dtype=np.uint32)
    np.testing.assert_array_equal(out.copy_to_host(), reference)
    for value, expected in ((1, 0xFFFFFFFF), (0, 0), (-1, 0)):
        uniform = cuda.to_device(np.full(64, value, np.int32))
        m.warp_vote_analyzer[1, 64](uniform, out)
        np.testing.assert_array_equal(out.copy_to_host(), np.full(2, expected, np.uint32))


def test_task20(gpu):
    cuda = gpu
    for n, threads in ((1, 1), (19, 23), (65, 128), (129, 256)):
        x = ((np.arange(n) % 9) - 4).astype(np.float32) / 4
        out = cuda.device_array_like(x)
        m.scan_hillis_steele[1, threads](cuda.to_device(x), out)
        np.testing.assert_allclose(out.copy_to_host(), np.cumsum(x), atol=1e-6)


def test_task21(gpu):
    cuda = gpu
    for n, threads in ((1, 1), (19, 32), (65, 128), (129, 256)):
        x = ((np.arange(n) % 7) - 3).astype(np.float32)
        out = cuda.device_array_like(x)
        m.scan_blelloch_work_efficient[1, threads](cuda.to_device(x), out)
        reference = np.cumsum(x)
        np.testing.assert_array_equal(out.copy_to_host(), reference)


def test_task22(gpu):
    cuda = gpu
    rng = np.random.default_rng(22)
    for n, bins in ((0, 1), (193, 7), (271, 256)):
        x = rng.integers(-2, bins + 2, n, dtype=np.int32)
        out = cuda.to_device(np.zeros(bins, dtype=np.int32))
        m.histogram_privatized[2, 32](cuda.to_device(x), out, bins)
        valid = x[(x >= 0) & (x < bins)]
        np.testing.assert_array_equal(out.copy_to_host(), np.bincount(valid, minlength=bins))


def test_task23(gpu):
    cuda = gpu
    for x in (np.array([-2, 0, 3, 5, -1, 7, 0, 3], np.int32),
              np.array([-1, 0, -3], np.int32),
              np.arange(-19, 54, dtype=np.int32)):
        out = cuda.to_device(np.full(x.size, -99, dtype=np.int32))
        count = cuda.to_device(np.zeros(1, dtype=np.int32))
        m.dynamic_work_queue[2, 16](cuda.to_device(x), out, count)
        size = int(count.copy_to_host()[0])
        assert size == np.count_nonzero(x > 0)
        actual = out.copy_to_host()
        np.testing.assert_array_equal(np.sort(actual[:size]), np.sort(x[x > 0]))
        np.testing.assert_array_equal(actual[size:], -99)


def test_task24(real_gpu):
    cuda = real_gpu
    x = np.array([11, -0.5, 7], dtype=np.float32)
    backing = cuda.to_device(x.view(np.int32))
    m.atomic_cas_float_add[3, 64](backing, 1, np.float32(0.25))
    result = backing.copy_to_host().view(np.float32)
    np.testing.assert_array_equal(result, np.array([11, 47.5, 7], np.float32))
    # Integer bit comparison must terminate even when the stored float is NaN.
    nan = cuda.to_device(np.array([np.nan], np.float32).view(np.int32))
    m.atomic_cas_float_add[1, 32](nan, 0, np.float32(1))
    assert np.isnan(nan.copy_to_host().view(np.float32)[0])


def test_task25(gpu):
    cuda = gpu
    rng = np.random.default_rng(25)
    dense = rng.normal(size=17).astype(np.float32)
    for n in (0, 1, 97):
        indices = rng.integers(0, dense.size, size=n, dtype=np.int32)
        values = rng.normal(size=n).astype(np.float32)
        result = cuda.to_device(np.zeros(1, np.float32))
        m.sparse_dot_product_atomic[2, 32](cuda.to_device(indices), cuda.to_device(values),
                                         cuda.to_device(dense), result)
        np.testing.assert_allclose(result.copy_to_host()[0],
                                   np.dot(values.astype(np.float64), dense[indices]),
                                   rtol=2e-5, atol=2e-5)