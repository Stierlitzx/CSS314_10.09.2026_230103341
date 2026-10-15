"""Real-device tests: streams, graph capture, managed memory, cooperative sync.

Functional references are NumPy; timings are checked for internal consistency,
not compared to invented speedup thresholds. Nsight is needed to certify overlap.
"""
import numpy as np
import pytest
from numba import cuda

from lab_kernels import module5_streams_cupy as m


@pytest.mark.parametrize("size,chunk", [(17, 32), (1031, 127), (0, 8)])
def test_task35_dual_stream(real_gpu, size, chunk):
    rng = np.random.default_rng(35)
    x = rng.normal(size=size).astype(np.float32)
    out = np.full_like(x, np.nan)
    assert m.pipeline_dual_stream(x, out, chunk) is out
    np.testing.assert_array_equal(out, x*2)


@pytest.mark.parametrize("streams", [1, 2, 4])
def test_task36_multistream_gemv(real_gpu, streams):
    rng = np.random.default_rng(36)
    a = rng.normal(size=(33, 29)).astype(np.float32)
    vectors = rng.normal(size=(29, 7)).astype(np.float32)
    out = np.full((33, 7), np.nan, np.float32)
    m.multistream_gemv(a, vectors, out, streams)
    np.testing.assert_allclose(out, a.astype(np.float64)@vectors, rtol=2e-5, atol=3e-6)


def test_task37_rawkernel_sinc(cupy_gpu):
    cp = cupy_gpu
    x = np.concatenate((np.array([0, 1, -1, 1e-7, -1e-7], np.float32),
                        np.linspace(-3.9, 4.1, 269, dtype=np.float32)))
    out = cp.empty(x.shape, cp.float32)
    m.cupy_rawkernel_sinc(cp.asarray(x), out)
    np.testing.assert_allclose(cp.asnumpy(out), np.sinc(x.astype(np.float64)), rtol=2e-5, atol=2e-7)


def test_task38_rawmodule_arithmetic(cupy_gpu):
    cp = cupy_gpu
    rng = np.random.default_rng(38)
    a, b = rng.normal(size=(2, 521)).astype(np.float32)
    outputs = [cp.empty_like(cp.asarray(a)) for _ in range(3)]
    m.cupy_rawmodule_arithmetic(cp.asarray(a), cp.asarray(b), *outputs)
    for got, expected in zip(outputs, (a+b, a-b, a*b)):
        np.testing.assert_allclose(cp.asnumpy(got), expected, rtol=2e-6, atol=2e-7)


@cuda.jit
def _profile_increment(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


def test_task39_event_profiler(real_gpu):
    x = real_gpu.to_device(np.zeros(1031, np.float32))
    stats = m.cuda_event_profiler(_profile_increment[5, 256], (x,), 2, 5)
    np.testing.assert_array_equal(x.copy_to_host(), np.full(1031, 7, np.float32))
    samples = np.asarray(stats["samples_ms"])
    assert samples.shape == (5,) and np.all(np.isfinite(samples)) and np.all(samples >= 0)
    assert stats["mean_ms"] == pytest.approx(samples.mean())
    assert stats["variance_ms2"] == pytest.approx(samples.var())
    assert stats["min_ms"] == samples.min()
    assert stats["max_ms"] == samples.max()


def test_task40_graph_capture(cupy_gpu):
    cp = cupy_gpu
    rng = np.random.default_rng(40)
    a, b = rng.normal(size=(2, 1031)).astype(np.float32)
    da, db, dc = cp.asarray(a), cp.asarray(b), cp.full(1031, cp.nan, dtype=cp.float32)
    m.cuda_graph_capture_exec(da, db, dc, 5)
    # Procedural reference for the documented two-node overwrite graph.
    ref = np.empty_like(a)
    for _ in range(5):
        np.add(a, b, out=ref)
        np.multiply(ref, 2, out=ref)
    np.testing.assert_allclose(cp.asnumpy(dc), ref, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(cp.asnumpy(da), a)
    np.testing.assert_array_equal(cp.asnumpy(db), b)


def test_task41_unified_memory(cupy_gpu):
    stats = m.benchmark_unified_memory(65536)
    assert stats["verified"]
    for key in ("first_touch_ms", "second_access_ms", "cpu_touch_seconds"):
        assert np.isfinite(stats[key]) and stats[key] >= 0
    if stats["prefetch_ms"] is None:
        assert stats["prefetch_unavailable_reason"]
    else:
        assert np.isfinite(stats["prefetch_ms"]) and stats["prefetch_ms"] >= 0


@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_task42_cooperative_barrier(cupy_gpu, iterations):
    cp = cupy_gpu
    x = np.arange(1031, dtype=np.float32) / 8
    x[0] = 2
    dx = cp.asarray(x)
    m.grid_cooperative_barrier(dx, iterations)
    ref = x.copy()
    for _ in range(iterations):
        ref += 1
        ref[1:] += ref[0]
    np.testing.assert_array_equal(cp.asnumpy(dx), ref)