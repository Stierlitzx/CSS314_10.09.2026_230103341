"""Tasks 26--34: scientific CUDA kernels (no CPU computational fallback).

Public functions use ``kernel[grid, block]`` except the Gaussian host launcher
and lazy-import CuPy Monte Carlo routine. Unless overridden, use <=256 threads
per block and enough blocks to cover the output. Arrays must not alias except
the explicitly in-place N-body and FFT operations. Float32/float64 are supported.
"""

import math
import operator

from numba import cuda, float64


@cuda.jit
def heat_diffusion_2d_step(u_curr, u_next, alpha, dt, dx, dy):
    """26: One explicit five-point diffusion step, with pinned/copied edges.

    Use 2D launch, e.g. block=(16,16), grid=(ceil(width/16),ceil(height/16)).
    dx,dy>0, dt>=0, alpha>=0; stability requires
    alpha*dt*(1/dx**2 + 1/dy**2) <= 1/2. Distinct equal-shaped arrays.
    """
    x, y = cuda.grid(2)
    h, w = u_curr.shape
    if x < w and y < h:
        value = u_curr[y, x]
        if 0 < x < w - 1 and 0 < y < h - 1:
            value += alpha * dt * (
                (u_curr[y, x - 1] - 2 * u_curr[y, x] + u_curr[y, x + 1]) / (dx * dx)
                + (u_curr[y - 1, x] - 2 * u_curr[y, x] + u_curr[y + 1, x]) / (dy * dy)
            )
        u_next[y, x] = value


@cuda.jit
def wave_equation_1d_step(u_prev, u_curr, u_next, c, dt, dx):
    """27: Centered second-order wave step; endpoints copied from u_curr.

    Distinct nonempty equal-length arrays, 1D blocks<=256. dx>0; CFL |c*dt/dx|<=1.
    Shared tile includes radius-one halos. All threads load before the barrier.
    """
    tile = cuda.shared.array(258, float64)
    i = cuda.grid(1)
    t = cuda.threadIdx.x
    n = u_curr.size
    tile[t + 1] = 0.0
    if i < n:
        tile[t + 1] = u_curr[i]
    if t == 0:
        tile[0] = u_curr[max(0, min(i - 1, n - 1))]
        tile[cuda.blockDim.x + 1] = u_curr[min(i + cuda.blockDim.x, n - 1)]
    cuda.syncthreads()
    if i < u_curr.size:
        value = tile[t + 1]
        if 0 < i < u_curr.size - 1:
            value = (2 * tile[t + 1] - u_prev[i] + (c * dt / dx) ** 2
                     * (tile[t] - 2 * tile[t + 1] + tile[t + 2]))
        u_next[i] = value


@cuda.jit
def nbody_all_pairs(positions, velocities, masses, dt, softening):
    """28: All-pairs gravitational velocity kick, G=1; positions UNCHANGED.

    positions,velocities: (N,3); masses: (N,). Acceleration is sum_j!=i
    m_j*(p_j-p_i)/(r**2+softening**2)**1.5. Require softening>0.
    Launch ceil(N/block) one-dimensional blocks, block<=256. Shared source-body
    tiles read immutable positions; one thread owns each velocity, so multiple
    blocks are safe. v_new=v_old+dt*a, no drift or global synchronization.
    """
    tile = cuda.shared.array((256, 3), float64)
    tile_mass = cuda.shared.array(256, float64)
    t = cuda.threadIdx.x
    n = positions.shape[0]
    block = cuda.blockDim.x
    i = cuda.grid(1)
    px, py, pz = 0.0, 0.0, 0.0
    if i < n:
        px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
    ax, ay, az = 0.0, 0.0, 0.0
    for source in range(0, n, block):
        j = source + t
        if j < n:
            tile[t, 0] = positions[j, 0]
            tile[t, 1] = positions[j, 1]
            tile[t, 2] = positions[j, 2]
            tile_mass[t] = masses[j]
        cuda.syncthreads()
        if i < n:
            for k in range(min(block, n - source)):
                if source + k != i:
                    rx = tile[k, 0] - px
                    ry = tile[k, 1] - py
                    rz = tile[k, 2] - pz
                    r2 = rx * rx + ry * ry + rz * rz + softening * softening
                    scale = tile_mass[k] / (r2 * math.sqrt(r2))
                    ax += rx * scale
                    ay += ry * scale
                    az += rz * scale
        cuda.syncthreads()
    if i < n:
        velocities[i, 0] += dt * ax
        velocities[i, 1] += dt * ay
        velocities[i, 2] += dt * az


@cuda.jit(device=True)
def _reflect_index(i, n):
    # SciPy ndimage mode='reflect': half-sample symmetry, including n=1.
    i %= 2 * n
    if i >= n:
        i = 2 * n - 1 - i
    return i


@cuda.jit
def _gaussian_horizontal(img_in, temp_buf, weights):
    tile = cuda.shared.array((16, 32), float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x0, y = cuda.blockIdx.x * 16, cuda.blockIdx.y * 16 + ty
    h, w = img_in.shape
    radius = weights.size // 2
    for s in range(tx, 16 + 2 * radius, 16):
        tile[ty, s] = img_in[_reflect_index(y, h), _reflect_index(x0 + s - radius, w)]
    cuda.syncthreads()
    x = x0 + tx
    if x < w and y < h:
        value = 0.0
        for k in range(weights.size):
            value += weights[k] * tile[ty, tx + 2 * radius - k]
        temp_buf[y, x] = value


@cuda.jit
def _gaussian_vertical(temp_buf, img_out, weights):
    tile = cuda.shared.array((32, 16), float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x, y0 = cuda.blockIdx.x * 16 + tx, cuda.blockIdx.y * 16
    h, w = temp_buf.shape
    radius = weights.size // 2
    for s in range(ty, 16 + 2 * radius, 16):
        tile[s, tx] = temp_buf[_reflect_index(y0 + s - radius, h), _reflect_index(x, w)]
    cuda.syncthreads()
    y = y0 + ty
    if x < w and y < h:
        value = 0.0
        for k in range(weights.size):
            value += weights[k] * tile[ty + 2 * radius - k, tx]
        img_out[y, x] = value


def gaussian_blur_2d_separable(img_in, img_out, temp_buf, kernel_1d):
    """29: Host launcher for two tiled CUDA convolution passes, reflect edges.

    Pass three distinct device arrays of identical nonempty 2D shape and a
    device 1D odd kernel of length 1..17 (radius<=8). Kernels launch 16x16
    threads and use halo tiles; no host convolution/copy or CPU fallback.
    Weights are used as supplied (not normalized). Convolution matches SciPy
    ndimage.convolve1d(..., mode='reflect'), horizontal then vertical. The
    intermediate is rounded to temp_buf's dtype. Operations use default stream.
    """
    if len(img_in.shape) != 2 or img_in.shape != img_out.shape or img_in.shape != temp_buf.shape:
        raise ValueError("input, output and temporary arrays must have identical 2D shapes")
    if min(img_in.shape) < 1:
        raise ValueError("image must be nonempty")
    if len(kernel_1d.shape) != 1 or not 1 <= kernel_1d.size <= 17 or kernel_1d.size % 2 == 0:
        raise ValueError("kernel must be 1D with odd length in [1,17]")
    h, w = img_in.shape
    grid = ((w + 15) // 16, (h + 15) // 16)
    _gaussian_horizontal[grid, (16, 16)](img_in, temp_buf, kernel_1d)
    _gaussian_vertical[grid, (16, 16)](temp_buf, img_out, kernel_1d)


@cuda.jit
def sobel_edge_detector(img_in, magnitude, orientation):
    """30: Sobel magnitude and atan2(gy,gx) orientation in radians.

    SciPy ndimage.sobel mode='nearest' (clamped), no 1/8 normalization. axis=1
    is gx, axis=0 gy. Nonempty equal shapes. Use 2D blocks<=16x16 (<=256
    threads); shared tile loads a one-pixel apron including block corners.
    """
    tile = cuda.shared.array((18, 18), float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by = cuda.blockDim.x, cuda.blockDim.y
    x, y = cuda.grid(2)
    h, w = img_in.shape
    for sy in range(ty, by + 2, by):
        for sx in range(tx, bx + 2, bx):
            iy = min(max(cuda.blockIdx.y * by + sy - 1, 0), h - 1)
            ix = min(max(cuda.blockIdx.x * bx + sx - 1, 0), w - 1)
            tile[sy, sx] = img_in[iy, ix]
    cuda.syncthreads()
    if x < w and y < h:
        gx, gy = 0.0, 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                value = tile[ty + dy + 1, tx + dx + 1]
                wx = dx * (2 if dy == 0 else 1)
                wy = dy * (2 if dx == 0 else 1)
                gx += wx * value
                gy += wy * value
        magnitude[y, x] = math.sqrt(gx * gx + gy * gy)
        orientation[y, x] = math.atan2(gy, gx)


def monte_carlo_pi_cupy(num_samples):
    """31: Estimate pi with genuine CuPy device RNG and reduction.

    CuPy is imported only on invocation; a CUDA-capable CuPy installation is
    required. There is no CPU fallback. Positive integer samples are processed
    in chunks <=2**20 to bound device memory. Returns a host float, synchronizing
    each chunk's reduction. Statistical output is not a deterministic benchmark.
    """
    num_samples = operator.index(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    import cupy as cp

    inside = 0
    remaining = num_samples
    while remaining:
        count = min(remaining, 1 << 20)
        samples = cp.random.random((count, 2), dtype=cp.float32)
        inside += int(cp.count_nonzero(cp.sum(samples * samples, axis=1) <= 1).get())
        remaining -= count
    return 4.0 * inside / num_samples


@cuda.jit
def kmeans_assign_clusters(points, centroids, labels, distances):
    """32: Assign each (N,D) point to nearest of K>=1 (K,D) centroids.

    distances contains SQUARED Euclidean distances; exact ties choose the
    lowest centroid index. One-dimensional blocks<=256, distinct outputs.
    K*D<=2048 centroid elements are cached cooperatively in shared float64;
    larger centroid tables use global memory, without a CPU fallback.
    """
    cache = cuda.shared.array(2048, float64)
    dimensions = points.shape[1]
    cached = centroids.shape[0] * dimensions <= 2048
    if cached:
        for j in range(cuda.threadIdx.x, centroids.shape[0] * dimensions, cuda.blockDim.x):
            cache[j] = centroids[j // dimensions, j % dimensions]
    cuda.syncthreads()
    i = cuda.grid(1)
    if i < points.shape[0]:
        best = math.inf
        label = 0
        for k in range(centroids.shape[0]):
            distance = 0.0
            for d in range(points.shape[1]):
                if cached:
                    center = cache[k * dimensions + d]
                else:
                    center = centroids[k, d]
                diff = points[i, d] - center
                distance += diff * diff
            if distance < best:
                best = distance
                label = k
        labels[i] = label
        distances[i] = best


@cuda.jit
def fft_radix2_butterfly_stage(data_re, data_im, stage, N):
    """33: One in-place, forward (negative-sign) radix-2 DIT FFT stage.

    N is power of two, arrays have length N, stage in [1,log2(N)]. Stage 1
    combines width-2 groups. Full FFT: BIT-REVERSE input first, then launch
    stages in ascending order on the same stream; output is unnormalized and
    natural order. Launch ceil((N/2)/block) 1D blocks, block<=256. Each butterfly
    owns a disjoint pair; arrays must not alias.
    """
    i = cuda.grid(1)
    if i < N // 2:
        width = 1 << stage
        half = width // 2
        j = i % half
        left = (i // half) * width + j
        right = left + half
        angle = -2.0 * math.pi * j / width
        wr, wi = math.cos(angle), math.sin(angle)
        rr, ri = data_re[right], data_im[right]
        tr, ti = wr * rr - wi * ri, wr * ri + wi * rr
        lr, li = data_re[left], data_im[left]
        data_re[left], data_im[left] = lr + tr, li + ti
        data_re[right], data_im[right] = lr - tr, li - ti


@cuda.jit
def mandelbrot_escape_time(out, xmin, xmax, ymin, ymax, max_iter):
    """34: Mandelbrot escape counts, z0=0; count first iterate with |z|>2.

    2D launch as heat; integer out of shape (height,width). Coordinates include
    both bounds; a singleton axis uses its minimum bound. Interior points get
    max_iter; max_iter>=0. One iteration is z=z*z+c then escape check.
    """
    x, y = cuda.grid(2)
    h, w = out.shape
    if x < w and y < h:
        cr, ci = xmin, ymin
        if w > 1:
            cr += (xmax - xmin) * x / (w - 1)
        if h > 1:
            ci += (ymax - ymin) * y / (h - 1)
        zr, zi = 0.0, 0.0
        count = 0
        while count < max_iter:
            zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
            count += 1
            if zr * zr + zi * zi > 4.0:
                break
        out[y, x] = count