"""NumPy/SciPy oracle checks for scientific CUDA tasks26--34."""

import numpy as np
from scipy import ndimage

from lab_kernels import module4_scientific as m


def _grid2(shape, block=(8, 8)):
    h, w = shape
    return ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])


def test_task26(gpu):
    cuda = gpu
    rng = np.random.default_rng(26)
    for shape in ((7, 11), (1, 5), (2, 3)):
        x = rng.normal(size=shape).astype(np.float32)
        alpha, dt, dx, dy = 0.2, 0.03, 0.7, 1.1
        ref = x.astype(np.float64).copy()
        ref[1:-1, 1:-1] += alpha * dt * (
            (x[1:-1, :-2] - 2 * x[1:-1, 1:-1] + x[1:-1, 2:]) / dx**2
            + (x[:-2, 1:-1] - 2 * x[1:-1, 1:-1] + x[2:, 1:-1]) / dy**2)
        out = cuda.device_array_like(x)
        m.heat_diffusion_2d_step[_grid2(shape), (8, 8)](
            cuda.to_device(x), out, alpha, dt, dx, dy)
        actual = out.copy_to_host()
        np.testing.assert_allclose(actual, ref, rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(actual[[0, -1]], x[[0, -1]])
        np.testing.assert_array_equal(actual[:, [0, -1]], x[:, [0, -1]])


def test_task27(gpu):
    cuda = gpu
    rng = np.random.default_rng(27)
    for n in (1, 2, 37):
        prev = rng.normal(size=n).astype(np.float32)
        curr = rng.normal(size=n).astype(np.float32)
        c, dt, dx = 1.3, 0.05, 0.4
        ref = curr.copy()
        ref[1:-1] = (2 * curr[1:-1] - prev[1:-1] + (c * dt / dx)**2
                     * (curr[:-2] - 2 * curr[1:-1] + curr[2:]))
        out = cuda.device_array_like(curr)
        m.wave_equation_1d_step[(n + 15) // 16, 16](
            cuda.to_device(prev), cuda.to_device(curr), out, c, dt, dx)
        actual = out.copy_to_host()
        np.testing.assert_allclose(actual, ref, rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(actual[[0, -1]], curr[[0, -1]])


def test_task28(gpu):
    cuda = gpu
    rng = np.random.default_rng(28)
    # More particles than threads exercises both source/target tile tails.
    for n in (1, 19):
        positions = rng.normal(size=(n, 3)).astype(np.float64)
        velocities = rng.normal(size=(n, 3)).astype(np.float64) / 10
        masses = rng.uniform(0.2, 1.2, size=n)
        dt, soft = 0.01, 0.2
        delta = positions[None, :, :] - positions[:, None, :]
        r2 = np.sum(delta * delta, axis=2) + soft**2
        accel = np.sum(delta * (masses[None, :] / r2**1.5)[:, :, None], axis=1)
        v_ref = velocities + dt * accel
        dp, dv = cuda.to_device(positions), cuda.to_device(velocities)
        m.nbody_all_pairs[(n + 7) // 8, 8](dp, dv, cuda.to_device(masses), dt, soft)
        np.testing.assert_allclose(dv.copy_to_host(), v_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(dp.copy_to_host(), positions)


def test_task29(gpu):
    cuda = gpu
    rng = np.random.default_rng(29)
    for shape, weights in (((5, 19), np.array([1, 4, 6, 4, 1], np.float32) / 16),
                           ((1, 3), np.array([1, 2, 3], np.float32) / 6),
                           ((3, 1), np.ones(17, np.float32) / 17)):
        x = rng.normal(size=shape).astype(np.float32)
        # Asymmetric weights additionally distinguish convolution/correlation.
        reference = ndimage.convolve1d(x, weights, axis=1, mode="reflect")
        reference = ndimage.convolve1d(reference, weights, axis=0, mode="reflect")
        out, temp = cuda.device_array_like(x), cuda.device_array_like(x)
        m.gaussian_blur_2d_separable(cuda.to_device(x), out, temp, cuda.to_device(weights))
        np.testing.assert_allclose(out.copy_to_host(), reference, rtol=3e-6, atol=3e-7)
    # Prescribed Gaussian sigma=2, truncate=4 => radius8 (17 coefficients).
    x = rng.normal(size=(7, 19))
    weights = np.exp(-np.arange(-8, 9, dtype=np.float64)**2 / 8)
    weights /= weights.sum()
    out, temp = cuda.device_array_like(x), cuda.device_array_like(x)
    m.gaussian_blur_2d_separable(cuda.to_device(x), out, temp, cuda.to_device(weights))
    np.testing.assert_allclose(out.copy_to_host(), ndimage.gaussian_filter(x, sigma=2, mode="reflect"),
                               rtol=2e-12, atol=2e-12)


def test_task30(gpu):
    cuda = gpu
    rng = np.random.default_rng(30)
    for shape in ((7, 13), (1, 5), (4, 1)):
        x = rng.normal(size=shape).astype(np.float64)
        gx = ndimage.sobel(x, axis=1, mode="nearest")
        gy = ndimage.sobel(x, axis=0, mode="nearest")
        magnitude, orientation = cuda.device_array_like(x), cuda.device_array_like(x)
        m.sobel_edge_detector[_grid2(shape), (8, 8)](cuda.to_device(x), magnitude, orientation)
        np.testing.assert_allclose(magnitude.copy_to_host(), np.hypot(gx, gy), rtol=2e-12, atol=2e-12)
        # Compare orientation modulo 2*pi, ignoring undefined near-zero gradients.
        valid = np.hypot(gx, gy) > 1e-10
        phase = orientation.copy_to_host() - np.arctan2(gy, gx)
        np.testing.assert_allclose(np.exp(1j * phase[valid]), 1, atol=2e-12)


def test_task31(cupy_gpu):
    cp = cupy_gpu
    n = 200_003
    # Reconstruct the exact GPU RNG stream with NumPy arithmetic as oracle.
    cp.random.seed(31)
    samples = cp.random.random((n, 2), dtype=cp.float32).get()
    reference = 4 * np.count_nonzero(np.sum(samples * samples, axis=1) <= 1) / n
    cp.random.seed(31)
    actual = m.monte_carlo_pi_cupy(n)
    assert actual == reference
    assert abs(actual - np.pi) < 0.04


def test_task32(gpu):
    cuda = gpu
    rng = np.random.default_rng(32)
    points = rng.normal(size=(37, 5)).astype(np.float32)
    centers = rng.normal(size=(4, 5)).astype(np.float32)
    centers[1] = centers[0]
    points[0] = centers[0]  # exact distance tie must choose index0
    squared = np.sum((points[:, None, :].astype(np.float64) - centers[None, :, :])**2, axis=2)
    reference = np.argmin(squared, axis=1)
    labels = cuda.device_array(points.shape[0], np.int32)
    distances = cuda.device_array(points.shape[0], np.float32)
    m.kmeans_assign_clusters[3, 16](cuda.to_device(points), cuda.to_device(centers), labels, distances)
    np.testing.assert_array_equal(labels.copy_to_host(), reference)
    np.testing.assert_allclose(distances.copy_to_host(), squared[np.arange(37), reference], rtol=2e-6)
    # K*D above2048 exercises device-global centroid path, not a host fallback.
    points = rng.normal(size=(3, 700)).astype(np.float32)
    centers = rng.normal(size=(3, 700)).astype(np.float32)
    squared = np.sum((points[:, None, :].astype(np.float64) - centers[None, :, :])**2, axis=2)
    reference = np.argmin(squared, axis=1)
    labels = cuda.device_array(3, np.int32)
    distances = cuda.device_array(3, np.float32)
    m.kmeans_assign_clusters[1, 16](cuda.to_device(points), cuda.to_device(centers), labels, distances)
    np.testing.assert_array_equal(labels.copy_to_host(), reference)
    np.testing.assert_allclose(distances.copy_to_host(), squared[np.arange(3), reference], rtol=2e-6)


def test_task33(gpu):
    cuda = gpu
    rng = np.random.default_rng(33)
    for n in (2, 8, 64):
        stages = n.bit_length() - 1
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        # DIT requires bit-reversed input BEFORE stage1 (which has width2).
        permutation = np.array([int(f"{i:0{stages}b}"[::-1], 2) for i in range(n)])
        reordered = x[permutation]
        re = cuda.to_device(reordered.real.copy())
        im = cuda.to_device(reordered.imag.copy())
        for stage in range(1, stages + 1):
            m.fft_radix2_butterfly_stage[(n // 2 + 15) // 16, 16](re, im, stage, n)
            if stage == 1:
                pairwise = reordered.copy()
                pairwise[::2] = reordered[::2] + reordered[1::2]
                pairwise[1::2] = reordered[::2] - reordered[1::2]
                np.testing.assert_allclose(re.copy_to_host() + 1j * im.copy_to_host(), pairwise, atol=1e-12)
        np.testing.assert_allclose(re.copy_to_host() + 1j * im.copy_to_host(), np.fft.fft(x),
                                   rtol=2e-12, atol=2e-12)


def test_task34(gpu):
    cuda = gpu
    for shape in ((9, 17), (1, 5), (4, 1)):
        h, w = shape
        xmin, xmax, ymin, ymax, limit = -2.1, 1.2, -1.3, 1.1, 31
        xx = np.linspace(xmin, xmax, w) if w > 1 else np.array([xmin])
        yy = np.linspace(ymin, ymax, h) if h > 1 else np.array([ymin])
        ref = np.zeros(shape, np.int32)
        for y in range(h):
            for x in range(w):
                zr, zi = 0.0, 0.0
                for iteration in range(1, limit + 1):
                    zr, zi = zr*zr - zi*zi + xx[x], 2*zr*zi + yy[y]
                    ref[y, x] = iteration
                    if zr*zr + zi*zi > 4:
                        break
        out = cuda.device_array(shape, np.int32)
        m.mandelbrot_escape_time[_grid2(shape), (8, 8)](out, xmin, xmax, ymin, ymax, limit)
        np.testing.assert_array_equal(out.copy_to_host(), ref)