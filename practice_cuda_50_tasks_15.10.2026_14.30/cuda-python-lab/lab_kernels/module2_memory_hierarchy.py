"""Tasks 9–17: global, shared, constant and pinned CUDA memory.

The required interfaces are bracket-launched Numba CUDA kernels,
except benchmark_pinned_transfer; benchmark_stride_bandwidth is an additional
host timing utility. Floating shared-memory tasks accept float32/float64 except the
bank-conflict-free transpose, which requires float32. Other scratch is float64
to avoid silently truncating double inputs. Buffers must not overlap.
Launch constraints are part of each kernel's contract, not auto-tuned defaults.
"""

import operator
from time import perf_counter

import numpy as np
from numba import config, cuda, float32, float64, int64


MAX_BLOCK_SIZE = 256
MAX_RADIUS = 32
MAX_FILTER_SIZE = 65
TILE_SIZE = 16
TRANSPOSE_TILE = 32
# Deliberately asymmetric: tests detect correlation/convolution reversal and
# the half-sample offset of an even-length "same" convolution.
WEIGHTS = np.arange(1, 65, dtype=np.float32) / np.float32(2080.0)
WEIGHTS.setflags(write=False)


@cuda.jit
def benchmark_strided_access(in_arr, out_arr, stride):
    """Task 09: out_arr[i]=in_arr[i*stride], with a covering 1D grid.

    stride is a positive integer; input length must be at least
    (out_arr.size-1)*stride+1 for nonempty output. This is a kernel, not a timer.
    """
    i = cuda.grid(1)
    if i < out_arr.size:
        out_arr[i] = in_arr[i * stride]


def benchmark_stride_bandwidth(num_outputs=1 << 20, num_repeats=10):
    """Task 09 optional host benchmark: event timings for strides 1,2,4,8,16,32.

    Requires real CUDA and positive integer counts; no CUDASIM timing. Returns
    a dict keyed by stride with median kernel ms and effective decimal GB/s
    (one float32 read plus write per output). This is useful-byte bandwidth,
    not measured DRAM traffic. Allocations, copies and JIT warmup are excluded.
    There is no assertion that any stride achieves a prescribed speed ratio.
    """
    num_outputs = operator.index(num_outputs)
    num_repeats = operator.index(num_repeats)
    if num_outputs <= 0 or num_repeats <= 0:
        raise ValueError("num_outputs and num_repeats must be positive integers")
    if config.ENABLE_CUDASIM:
        raise RuntimeError("Stride bandwidth timing requires real CUDA, not CUDASIM")
    host = np.arange((num_outputs - 1) * 32 + 1, dtype=np.float32)
    source = cuda.to_device(host)
    dest = cuda.device_array(num_outputs, dtype=np.float32)
    blocks = (num_outputs + 255) // 256
    start, end = cuda.event(timing=True), cuda.event(timing=True)
    result = {}
    for stride in (1, 2, 4, 8, 16, 32):
        benchmark_strided_access[blocks, 256](source, dest, stride)
        cuda.synchronize()
        samples = []
        for _ in range(num_repeats):
            start.record()
            benchmark_strided_access[blocks, 256](source, dest, stride)
            end.record()
            end.synchronize()
            samples.append(cuda.event_elapsed_time(start, end))
        elapsed = float(np.median(samples))
        if elapsed <= 0:
            raise RuntimeError("CUDA event resolution insufficient; increase num_outputs")
        np.testing.assert_array_equal(dest.copy_to_host(), host[:num_outputs * stride:stride])
        result[stride] = {"kernel_ms": elapsed, "effective_gbps": num_outputs * 8 / (elapsed * 1e6)}
    return result


@cuda.jit
def stencil_1d_shared(in_arr, out_arr, radius):
    """Task 10: moving average with clamped boundaries and shared halos.

    Equal input/output lengths; integer 0<=radius<=32. Launch 1D blocks of
    1..256 threads, covering input. Cooperative halo loading also works when
    radius exceeds blockDim.x. Every thread participates in the barrier.
    """
    # Explicit literal: some CUDA compiler versions do not literal-fold
    # arithmetic on module constants in shared.array shapes. 256 + 2*32.
    tile = cuda.shared.array(320, float64)
    tx = cuda.threadIdx.x
    start = cuda.blockIdx.x * cuda.blockDim.x
    width = cuda.blockDim.x + 2 * radius
    for j in range(tx, width, cuda.blockDim.x):
        src = start + j - radius
        if in_arr.size:
            src = max(0, min(src, in_arr.size - 1))
            tile[j] = in_arr[src]
        else:
            tile[j] = 0.0
    cuda.syncthreads()
    i = start + tx
    if i < out_arr.size:
        value = 0.0
        for k in range(2 * radius + 1):
            value += tile[tx + k]
        out_arr[i] = value / (2 * radius + 1)


@cuda.jit
def conv_1d_shared_tiled(x, kernel, y):
    """Task 11: true convolution, signal-length centered 'same' output.

    Launch covering 1D blocks of 1..256 threads; 1<=kernel.size<=65.
    y.size=x.size, with zero padding. For x.size>=kernel.size this matches
    np.convolve(x,kernel,'same'), including even kernels. For shorter x it is
    full_convolution[(K-1)//2:(K-1)//2+x.size]. No in-place operation.
    """
    # 256 + 65 - 1; keep allocation dimensions literal for CUDA typing.
    tile = cuda.shared.array(320, float64)
    shared_filter = cuda.shared.array(65, float64)
    tx = cuda.threadIdx.x
    start = cuda.blockIdx.x * cuda.blockDim.x
    ksize = kernel.size
    for k in range(tx, ksize, cuda.blockDim.x):
        shared_filter[k] = kernel[k]
    left = ksize // 2
    for j in range(tx, cuda.blockDim.x + ksize - 1, cuda.blockDim.x):
        src = start + j - left
        tile[j] = x[src] if 0 <= src < x.size else 0.0
    cuda.syncthreads()
    i = start + tx
    if i < y.size:
        value = 0.0
        for k in range(ksize):
            value += tile[tx + k] * shared_filter[ksize - 1 - k]
        y[i] = value


@cuda.jit
def matmul_tiled_shared(A, B, C):
    """Task 12: C=A@B, rectangular matrices with matching inner dimension.

    Requires block=(16,16), grid=(ceil(N/16),ceil(M/16)) for A(M,K), B(K,N).
    Partial tiles and K=0 are supported; output cannot alias either input.
    """
    sa = cuda.shared.array((16, 16), float64)
    sb = cuda.shared.array((16, 16), float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    col, row = cuda.grid(2)
    value = 0.0
    for base in range(0, A.shape[1], TILE_SIZE):
        sa[ty, tx] = A[row, base + tx] if row < A.shape[0] and base + tx < A.shape[1] else 0.0
        sb[ty, tx] = B[base + ty, col] if base + ty < B.shape[0] and col < B.shape[1] else 0.0
        cuda.syncthreads()
        for k in range(TILE_SIZE):
            value += sa[ty, k] * sb[k, tx]
        cuda.syncthreads()
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = value


@cuda.jit
def transpose_bank_conflict_free(A, B):
    """Task 13: B=A.T using a padded 32x33 shared tile.

    Requires block=(32,8) or (32,32), grid=(ceil(cols/32),ceil(rows/32)).
    Input and output must be float32. Each thread loops over tile rows, so
    rectangular/partial tiles are safe. The 33-float pitch maps column-wise
    accesses onto distinct 32-bit banks (not a float64 performance claim).
    """
    tile = cuda.shared.array((32, 33), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x = cuda.blockIdx.x * TRANSPOSE_TILE + tx
    y = cuda.blockIdx.y * TRANSPOSE_TILE + ty
    for j in range(0, TRANSPOSE_TILE, cuda.blockDim.y):
        if y + j < A.shape[0] and x < A.shape[1]:
            tile[ty + j, tx] = A[y + j, x]
    cuda.syncthreads()
    out_col = cuda.blockIdx.y * TRANSPOSE_TILE + tx
    out_row = cuda.blockIdx.x * TRANSPOSE_TILE + ty
    for j in range(0, TRANSPOSE_TILE, cuda.blockDim.y):
        if out_row + j < B.shape[0] and out_col < B.shape[1]:
            B[out_row + j, out_col] = tile[tx, ty + j]


@cuda.jit
def reduction_sum_shared(in_arr, block_sums):
    """Task 14: one sum per 256-element block; launch exactly 256 threads.

    grid=ceil(N/256), block_sums.size>=grid. Empty input may use one block
    producing zero. Caller finishes reduction of these partial sums on host
    or in a subsequent launch. All 256 threads reach every barrier.
    """
    scratch = cuda.shared.array(256, float64)
    tx = cuda.threadIdx.x
    i = cuda.grid(1)
    scratch[tx] = in_arr[i] if i < in_arr.size else 0.0
    cuda.syncthreads()
    step = MAX_BLOCK_SIZE // 2
    while step:
        if tx < step:
            scratch[tx] += scratch[tx + step]
        cuda.syncthreads()
        step //= 2
    if tx == 0:
        block_sums[cuda.blockIdx.x] = scratch[0]


@cuda.jit
def reduction_argmax_shared(in_arr, block_max_vals, block_max_indices):
    """Task 15: per-block max and global index; exactly 256 threads/block.

    grid=ceil(N/256), outputs >=grid; ties select smallest index. Input must
    not contain NaNs; infinities are supported. Empty blocks yield (-inf,-1).
    Caller performs the final cross-block selection with the same tie rule.
    """
    vals = cuda.shared.array(256, float64)
    indices = cuda.shared.array(256, int64)
    tx = cuda.threadIdx.x
    i = cuda.grid(1)
    vals[tx] = in_arr[i] if i < in_arr.size else -np.inf
    indices[tx] = i if i < in_arr.size else -1
    cuda.syncthreads()
    step = MAX_BLOCK_SIZE // 2
    while step:
        if tx < step:
            other = tx + step
            if indices[other] >= 0 and (indices[tx] < 0 or vals[other] > vals[tx]
                    or (vals[other] == vals[tx] and indices[other] < indices[tx])):
                vals[tx] = vals[other]
                indices[tx] = indices[other]
        cuda.syncthreads()
        step //= 2
    if tx == 0:
        block_max_vals[cuda.blockIdx.x] = vals[0]
        block_max_indices[cuda.blockIdx.x] = indices[0]


@cuda.jit
def constant_memory_filter(signal, out):
    """Task 16: fixed 64-tap FIR convolution using exported constant WEIGHTS.

    Use any 1D covering launch, out.size=signal.size, zero padding. For N>=64
    matches np.convolve(signal,WEIGHTS,'same'); for N<64 returns the centered
    signal-length slice full_convolution[31:31+N], not NumPy's max(N,64) output.
    """
    weights = cuda.const.array_like(WEIGHTS)
    i = cuda.grid(1)
    if i < out.size:
        value = 0.0
        for k in range(64):
            src = i + 31 - k
            if 0 <= src < signal.size:
                value += signal[src] * weights[k]
        out[i] = value


def benchmark_pinned_transfer(size_bytes):
    """Task 17: measure actual pageable/pinned H2D and D2H transfers.

    Host utility (not bracket-launched), requiring real CUDA; CUDASIM raises
    RuntimeError rather than reporting fictional bandwidth. size_bytes must
    be a positive integer. Returns median CUDA-event milliseconds over five
    synchronized copies per direction, decimal GB/s, and H2D/D2H speedups.
    Also returns wall-clock milliseconds including Python/driver overhead.
    Allocation and warmup are excluded. This measures synchronous copies, not
    overlap, and promises no speedup; pageable staging may affect event timing.
    """
    size_bytes = operator.index(size_bytes)
    if size_bytes <= 0:
        raise ValueError("size_bytes must be a positive integer")
    if config.ENABLE_CUDASIM:
        raise RuntimeError("Pinned-memory transfer timing requires real CUDA, not CUDASIM")
    pageable = np.full(size_bytes, 37, dtype=np.uint8)
    pinned = cuda.pinned_array(size_bytes, dtype=np.uint8)
    pinned[:] = pageable
    device = cuda.device_array(size_bytes, dtype=np.uint8)

    def measure(copy):
        copy()
        cuda.synchronize()
        event_samples, wall_samples = [], []
        start_event, end_event = cuda.event(timing=True), cuda.event(timing=True)
        for _ in range(5):
            cuda.synchronize()
            start = perf_counter()
            start_event.record()
            copy()
            end_event.record()
            end_event.synchronize()
            event_samples.append(cuda.event_elapsed_time(start_event, end_event))
            wall_samples.append((perf_counter() - start) * 1000.0)
        return float(np.median(event_samples)), float(np.median(wall_samples))

    result = {"size_bytes": size_bytes, "repetitions": 5}
    copies = {
        "pageable_h2d": lambda: device.copy_to_device(pageable),
        "pinned_h2d": lambda: device.copy_to_device(pinned),
        "pageable_d2h": lambda: device.copy_to_host(pageable),
        "pinned_d2h": lambda: device.copy_to_host(pinned),
    }
    for name, copy in copies.items():
        if name.endswith("h2d"):
            # Distinct prior contents make a missing/wrong transfer detectable.
            device.copy_to_device(np.zeros(size_bytes, dtype=np.uint8))
        elif name == "pageable_d2h":
            pageable.fill(0)
        else:
            pinned.fill(0)
        elapsed, wall_elapsed = measure(copy)
        if elapsed <= 0:
            raise RuntimeError("CUDA event timing resolution was insufficient; increase size_bytes")
        result[name + "_ms"] = elapsed
        result[name + "_wall_ms"] = wall_elapsed
        result[name + "_gbps"] = size_bytes / (elapsed * 1e6)
        if name.endswith("h2d") and not np.all(device.copy_to_host() == 37):
            raise RuntimeError("Host-to-device transfer verification failed")
    if not (np.all(pageable == 37) and np.all(pinned == 37)):
        raise RuntimeError("Transfer verification failed")
    for direction in ("h2d", "d2h"):
        result[direction + "_speedup"] = (result["pageable_" + direction + "_ms"]
                                            / result["pinned_" + direction + "_ms"])
    return result