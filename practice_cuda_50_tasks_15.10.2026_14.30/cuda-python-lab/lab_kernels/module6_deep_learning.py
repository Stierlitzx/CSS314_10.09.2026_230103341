"""Tasks 43--50: real CUDA kernels (Numba bracket-launch API).

All inputs are finite, C-contiguous float32 device arrays unless stated otherwise.
43: [rows,256], logits/probs (rows,D), D>0. 44: 1D grid, flat or shaped
    x/out; bias (D,) repeats every D consecutive elements.
45: 1D grid over N*C*OH*OW, NCHW valid pooling, int32/64 indices
    are flattened *spatial* indices, with earliest index winning ties.
46: [features,256], x/out (B,features), population variance, B>0.
47: [rows,256], (rows,D), D>0, true warp Welford tuple reductions;
    requires hardware shuffle support and is not CUDA-simulator compatible.
48: arbitrary 1D grid, CSR with int32/64 indices, one thread per row.
49: [Nq,128], Q(Nq,D), K(Nk,D), V(Nk,Dv), Out(Nq,Dv).
    1<=D,Dv<=128, Nk>0, 16-key shared tiles, online softmax; no NxN
    intermediate. Batch/head dimensions can be processed by separate launches.
50: [B*S*H,128], q/q_rot (B,S,H,D), D even, 2<=D<=256;
    trig tables (S,D/2), adjacent rotary pairs, shared table cache per head.
No host computation fallback is used by any kernel.
"""
import math

from numba import cuda, float32, float64, int32


@cuda.jit
def stable_online_softmax(logits, probs):
    row = cuda.blockIdx.x
    t = cuda.threadIdx.x
    maximum = cuda.shared.array(256, float32)
    totals = cuda.shared.array(256, float32)
    m = -math.inf
    for j in range(t, logits.shape[1], 256):
        m = max(m, logits[row, j])
    maximum[t] = m
    cuda.syncthreads()
    step = 128
    while step:
        if t < step:
            maximum[t] = max(maximum[t], maximum[t + step])
        cuda.syncthreads()
        step //= 2
    s = float32(0)
    for j in range(t, logits.shape[1], 256):
        s += math.exp(logits[row, j] - maximum[0])
    totals[t] = s
    cuda.syncthreads()
    step = 128
    while step:
        if t < step:
            totals[t] += totals[t + step]
        cuda.syncthreads()
        step //= 2
    for j in range(t, logits.shape[1], 256):
        probs[row, j] = math.exp(logits[row, j] - maximum[0]) / totals[0]


@cuda.jit
def fused_bias_gelu(x, bias, out):
    for i in range(cuda.grid(1), x.size, cuda.gridsize(1)):
        z = x.flat[i] + bias[i % bias.size]
        out.flat[i] = 0.5 * z * (1.0 + math.tanh(0.7978845608028654 *
                                              (z + 0.044715 * z * z * z)))


@cuda.jit
def maxpool2d_with_argmax(x, out, argmax_idx, pool_h, pool_w, stride):
    for i in range(cuda.grid(1), out.size, cuda.gridsize(1)):
        ox = i % out.shape[3]
        oy = (i // out.shape[3]) % out.shape[2]
        c = (i // (out.shape[3] * out.shape[2])) % x.shape[1]
        n = i // (out.shape[3] * out.shape[2] * x.shape[1])
        best = -math.inf
        idx = oy * stride * x.shape[3] + ox * stride
        for dy in range(pool_h):
            for dx in range(pool_w):
                iy, ix = oy * stride + dy, ox * stride + dx
                v = x[n, c, iy, ix]
                if v > best:
                    best = v
                    idx = iy * x.shape[3] + ix
        out[n, c, oy, ox] = best
        argmax_idx[n, c, oy, ox] = idx


@cuda.jit
def batch_norm_1d_forward(x, gamma, beta, eps, out, mean_out, var_out):
    c, t = cuda.blockIdx.x, cuda.threadIdx.x
    scratch = cuda.shared.array(256, float32)
    mu = cuda.shared.array(1, float32)
    v = float32(0)
    for r in range(t, x.shape[0], 256):
        v += x[r, c]
    scratch[t] = v
    cuda.syncthreads()
    step = 128
    while step:
        if t < step:
            scratch[t] += scratch[t + step]
        cuda.syncthreads()
        step //= 2
    if t == 0:
        mu[0] = scratch[0] / x.shape[0]
        mean_out[c] = mu[0]
    cuda.syncthreads()
    v = float32(0)
    for r in range(t, x.shape[0], 256):
        d = x[r, c] - mu[0]
        v += d * d
    scratch[t] = v
    cuda.syncthreads()
    step = 128
    while step:
        if t < step:
            scratch[t] += scratch[t + step]
        cuda.syncthreads()
        step //= 2
    variance = scratch[0] / x.shape[0]
    if t == 0:
        var_out[c] = variance
    for r in range(t, x.shape[0], 256):
        out[r, c] = (x[r, c] - mu[0]) / math.sqrt(variance + eps) * gamma[c] + beta[c]


@cuda.jit(device=True)
def _warp_welford(n, mean, m2):
    lane = cuda.threadIdx.x % 32
    delta = 16
    while delta:
        nb = cuda.shfl_down_sync(0xffffffff, n, delta)
        mb = cuda.shfl_down_sync(0xffffffff, mean, delta)
        sb = cuda.shfl_down_sync(0xffffffff, m2, delta)
        if lane + delta < 32 and nb > 0:
            total = n + nb
            diff = mb - mean
            m2 += sb + diff * diff * n * nb / total
            mean += diff * nb / total
            n = total
        delta //= 2
    return n, mean, m2


@cuda.jit
def layer_norm_welford(x, gamma, beta, eps, out):
    t, row = cuda.threadIdx.x, cuda.blockIdx.x
    counts = cuda.shared.array(8, int32)
    # Float64 tuples avoid cancellation for large offsets with small variance.
    # Warp shuffles transport both halves of a float64 value on real hardware.
    means = cuda.shared.array(8, float64)
    sums = cuda.shared.array(8, float64)
    n, mu, m2 = int32(0), float64(0), float64(0)
    for j in range(t, x.shape[1], 256):
        n += 1
        d = x[row, j] - mu
        mu += d / n
        m2 += d * (x[row, j] - mu)
    n, mu, m2 = _warp_welford(n, mu, m2)
    if t % 32 == 0:
        counts[t // 32], means[t // 32], sums[t // 32] = n, mu, m2
    cuda.syncthreads()
    if t < 32:
        n, mu, m2 = int32(0), float64(0), float64(0)
        if t < 8:
            n, mu, m2 = counts[t], means[t], sums[t]
        n, mu, m2 = _warp_welford(n, mu, m2)
        if t == 0:
            means[0], sums[0] = mu, m2 / n
    cuda.syncthreads()
    for j in range(t, x.shape[1], 256):
        out[row, j] = (x[row, j] - means[0]) / math.sqrt(sums[0] + eps) * gamma[j] + beta[j]


@cuda.jit
def spmv_csr_kernel(row_ptr, col_indices, values, x, y, num_rows):
    for row in range(cuda.grid(1), num_rows, cuda.gridsize(1)):
        acc = float32(0)
        for k in range(row_ptr[row], row_ptr[row + 1]):
            acc += values[k] * x[col_indices[k]]
        y[row] = acc


@cuda.jit
def attention_sdpa_tiled(Q, K, V, Out, scale):
    row, t = cuda.blockIdx.x, cuda.threadIdx.x
    keys = cuda.shared.array((16, 128), float32)
    vals = cuda.shared.array((16, 128), float32)
    weights = cuda.shared.array(16, float32)
    state = cuda.shared.array(3, float32)  # running max, sum, old-output factor
    if t == 0:
        state[0], state[1] = -math.inf, 0
    acc = float32(0)
    cuda.syncthreads()
    for start in range(0, K.shape[0], 16):
        for i in range(t, 16 * Q.shape[1], 128):
            k, d = i // Q.shape[1], i % Q.shape[1]
            keys[k, d] = K[start + k, d] if start + k < K.shape[0] else 0
        for i in range(t, 16 * V.shape[1], 128):
            k, d = i // V.shape[1], i % V.shape[1]
            vals[k, d] = V[start + k, d] if start + k < K.shape[0] else 0
        cuda.syncthreads()
        if t < 16:
            score = float32(0)
            for d in range(Q.shape[1]):
                score += Q[row, d] * keys[t, d]
            weights[t] = score * scale if start + t < K.shape[0] else -math.inf
        cuda.syncthreads()
        if t == 0:
            tile_max = -math.inf
            for k in range(16):
                tile_max = max(tile_max, weights[k])
            new_max = max(state[0], tile_max)
            alpha = math.exp(state[0] - new_max)
            total = state[1] * alpha
            for k in range(16):
                weights[k] = math.exp(weights[k] - new_max)
                total += weights[k]
            state[0], state[1], state[2] = new_max, total, alpha
        cuda.syncthreads()
        if t < V.shape[1]:
            acc *= state[2]
            for k in range(16):
                acc += weights[k] * vals[k, t]
        cuda.syncthreads()
    if t < V.shape[1]:
        Out[row, t] = acc / state[1]


@cuda.jit
def fused_rope_embedding(q, cos_table, sin_table, q_rot):
    t, head = cuda.threadIdx.x, cuda.blockIdx.x
    h = head % q.shape[2]
    s = (head // q.shape[2]) % q.shape[1]
    b = head // (q.shape[1] * q.shape[2])
    cs = cuda.shared.array(128, float32)
    sn = cuda.shared.array(128, float32)
    for j in range(t, q.shape[3] // 2, cuda.blockDim.x):
        cs[j], sn[j] = cos_table[s, j], sin_table[s, j]
    cuda.syncthreads()
    for j in range(t, q.shape[3] // 2, cuda.blockDim.x):
        a, bval = q[b, s, h, 2 * j], q[b, s, h, 2 * j + 1]
        q_rot[b, s, h, 2 * j] = a * cs[j] - bval * sn[j]
        q_rot[b, s, h, 2 * j + 1] = a * sn[j] + bval * cs[j]