"""Independent NumPy/SciPy references; only warp Welford needs real hardware."""
import numpy as np
import pytest
from scipy.special import softmax
from scipy.sparse import csr_matrix

from lab_kernels import module6_deep_learning as m


def test_task43_softmax(gpu):
    rng = np.random.default_rng(43)
    x = (rng.normal(size=(3, 269)) * 30 + 1000).astype(np.float32)
    x[1] = -1000
    dx, dy = gpu.to_device(x), gpu.device_array_like(x)
    m.stable_online_softmax[3, 256](dx, dy)
    got = dy.copy_to_host()
    np.testing.assert_allclose(got, softmax(x.astype(np.float64), axis=1), rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(got.sum(axis=1), 1, atol=2e-6)


@pytest.mark.parametrize("flat", [False, True])
def test_task44_bias_gelu(gpu, flat):
    rng = np.random.default_rng(44)
    x = rng.normal(size=(3, 19)).astype(np.float32)
    bias = rng.normal(size=19).astype(np.float32)
    xx = x.ravel() if flat else x
    out = gpu.device_array_like(xx)
    m.fused_bias_gelu[2, 32](gpu.to_device(xx), gpu.to_device(bias), out)
    z = x.astype(np.float64) + bias
    ref = .5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + .044715 * z**3)))
    np.testing.assert_allclose(out.copy_to_host().reshape(x.shape), ref, rtol=2e-5, atol=3e-7)


def test_task45_maxpool_argmax(gpu):
    rng = np.random.default_rng(45)
    x = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    x[0, 0, :3, :2] = 5  # ties must select the earliest spatial index
    out = gpu.device_array((1, 2, 3, 4), np.float32)
    indices = gpu.device_array((1, 2, 3, 4), np.int64)
    m.maxpool2d_with_argmax[3, 16](gpu.to_device(x), out, indices, 3, 2, 2)
    ref, ri = np.empty(out.shape, np.float32), np.empty(indices.shape, np.int64)
    for n, c, iy, ix in np.ndindex(ref.shape):
        patch = x[n, c, iy*2:iy*2+3, ix*2:ix*2+2]
        dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
        ref[n, c, iy, ix] = patch[dy, dx]
        ri[n, c, iy, ix] = (iy*2+dy)*x.shape[3] + ix*2+dx
    np.testing.assert_array_equal(out.copy_to_host(), ref)
    np.testing.assert_array_equal(indices.copy_to_host(), ri)


def test_task46_batch_norm(gpu):
    rng = np.random.default_rng(46)
    x = rng.normal(size=(17, 3)).astype(np.float32)
    x[:, 1] = 3
    gamma = np.array([1.2, -2, .7], np.float32)
    beta = np.array([-.2, .5, 1], np.float32)
    out, mu, var = gpu.device_array_like(x), gpu.device_array(3, np.float32), gpu.device_array(3, np.float32)
    m.batch_norm_1d_forward[3, 256](gpu.to_device(x), gpu.to_device(gamma), gpu.to_device(beta),
                                         1e-5, out, mu, var)
    mean = x.astype(np.float64).mean(axis=0)
    variance = x.astype(np.float64).var(axis=0)
    np.testing.assert_allclose(mu.copy_to_host(), mean, atol=3e-7)
    np.testing.assert_allclose(var.copy_to_host(), variance, atol=3e-7)
    np.testing.assert_allclose(out.copy_to_host(), (x-mean)/np.sqrt(variance+1e-5)*gamma+beta,
                               rtol=3e-6, atol=5e-7)


@pytest.mark.parametrize("width", [7, 269, 1027])
def test_task47_layer_norm_welford(real_gpu, width):
    gpu = real_gpu
    rng = np.random.default_rng(47)
    x = (rng.normal(size=(3, width)) * .2 + 10000).astype(np.float32)
    x[1] = 10000
    gamma = rng.normal(size=width).astype(np.float32)
    beta = rng.normal(size=width).astype(np.float32)
    out = gpu.device_array_like(x)
    m.layer_norm_welford[3, 256](gpu.to_device(x), gpu.to_device(gamma), gpu.to_device(beta), 1e-5, out)
    xd = x.astype(np.float64)
    ref = (xd-xd.mean(axis=1, keepdims=True))/np.sqrt(xd.var(axis=1, keepdims=True)+1e-5)*gamma+beta
    np.testing.assert_allclose(out.copy_to_host(), ref, rtol=1e-6, atol=1e-5)


def test_task48_spmv(gpu):
    # Includes an empty row, empty final row, unsorted columns, and rectangular shape.
    ptr = np.array([0, 2, 2, 5, 6, 6], np.int32)
    cols = np.array([4, 0, 3, 1, 5, 2], np.int32)
    vals = np.array([2, -1, 3, .5, -2, 4], np.float32)
    x = np.arange(6, dtype=np.float32) / 3
    out = gpu.device_array(5, np.float32)
    m.spmv_csr_kernel[1, 32](gpu.to_device(ptr), gpu.to_device(cols), gpu.to_device(vals),
                                    gpu.to_device(x), out, 5)
    ref = csr_matrix((vals, cols, ptr), shape=(5, 6)) @ x
    np.testing.assert_allclose(out.copy_to_host(), ref, atol=3e-7)


@pytest.mark.parametrize("shape", [(2, 19, 7, 5), (1, 3, 64, 128)])
def test_task49_tiled_attention(gpu, shape):
    nq, nk, d, dv = shape
    rng = np.random.default_rng(49)
    q = (rng.normal(size=(nq, d))*3).astype(np.float32)
    k = (rng.normal(size=(nk, d))*3).astype(np.float32)
    v = rng.normal(size=(nk, dv)).astype(np.float32)
    out = gpu.device_array((nq, dv), np.float32)
    scale = np.float32(1/np.sqrt(d))
    m.attention_sdpa_tiled[nq, 128](gpu.to_device(q), gpu.to_device(k), gpu.to_device(v), out, scale)
    ref = softmax((q.astype(np.float64) @ k.astype(np.float64).T)*scale, axis=1) @ v
    np.testing.assert_allclose(out.copy_to_host(), ref, rtol=5e-5, atol=3e-6)


def test_task50_rope(gpu):
    rng = np.random.default_rng(50)
    q = rng.normal(size=(1, 3, 2, 10)).astype(np.float32)
    theta = rng.normal(size=(3, 5)).astype(np.float32)
    cs, sn = np.cos(theta), np.sin(theta)
    out = gpu.device_array_like(q)
    m.fused_rope_embedding[6, 128](gpu.to_device(q), gpu.to_device(cs), gpu.to_device(sn), out)
    ref = np.empty_like(q)
    ref[..., 0::2] = q[..., 0::2] * cs[None, :, None, :] - q[..., 1::2] * sn[None, :, None, :]
    ref[..., 1::2] = q[..., 0::2] * sn[None, :, None, :] + q[..., 1::2] * cs[None, :, None, :]
    got = out.copy_to_host()
    np.testing.assert_allclose(got, ref, rtol=2e-6, atol=3e-7)
    np.testing.assert_allclose((got**2).sum(axis=-1), (q**2).sum(axis=-1), rtol=2e-6)