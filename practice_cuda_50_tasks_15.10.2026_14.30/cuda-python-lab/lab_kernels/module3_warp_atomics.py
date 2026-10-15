"""Tasks 18--25: warp collectives, single-block scans, and atomics.

All public functions are Numba CUDA kernels, invoked with ``kernel[grid, block]``.
Unless stated otherwise use one-dimensional blocks of at most 256 threads.
Inputs and outputs must not alias, except the explicitly in-place CAS operation.
Warp intrinsics and libdevice bitcasts require real CUDA, not the CUDA simulator.
"""

from numba import cuda, float64, int32
from numba.cuda import libdevice


@cuda.jit
def warp_reduce_sum_shfl(in_arr, out_arr):
    """18: Shuffle-reduce warps, then aggregate to ONE sum per block.

    Launch ceil(n / blockDim.x) blocks, blockDim.x a multiple of 32 <=256.
    Allocate out_arr with gridDim.x elements. Out-of-range input lanes add zero.
    Every physical lane participates in the full-mask shuffle. Warp leaders
    publish partial sums to shared memory; the first warp reduces those sums.
    """
    partial = cuda.shared.array(8, float64)
    i = cuda.grid(1)
    lane = cuda.threadIdx.x % 32
    value = 0.0
    if i < in_arr.size:
        value = in_arr[i]
    offset = 16
    while offset:
        value += cuda.shfl_down_sync(0xFFFFFFFF, value, offset)
        offset //= 2
    if lane == 0:
        partial[cuda.threadIdx.x // 32] = value
    cuda.syncthreads()
    if cuda.threadIdx.x < 32:
        value = 0.0
        if lane < cuda.blockDim.x // 32:
            value = partial[lane]
        offset = 16
        while offset:
            value += cuda.shfl_down_sync(0xFFFFFFFF, value, offset)
            offset //= 2
        if lane == 0:
            out_arr[cuda.blockIdx.x] = value


@cuda.jit
def warp_vote_analyzer(data, mask_out):
    """19: Ballot the predicate data[i] > 0 into one uint32 mask per warp.

    Launch blocks of a multiple of 32 threads<=256; allocate one uint32 per
    physical warp, not per block. Out-of-range lanes vote false, so
    bits beyond the tail are zero. Use uint32 output to preserve bit 31.
    """
    i = cuda.grid(1)
    positive = False
    if i < data.size:
        positive = data[i] > 0
    mask = cuda.ballot_sync(0xFFFFFFFF, positive)
    all_positive = cuda.all_sync(0xFFFFFFFF, positive)
    any_positive = cuda.any_sync(0xFFFFFFFF, positive)
    if cuda.threadIdx.x % 32 == 0:
        if all_positive:
            mask_out[i // 32] = 0xFFFFFFFF
        elif any_positive:
            mask_out[i // 32] = mask
        else:
            mask_out[i // 32] = 0


@cuda.jit
def scan_hillis_steele(x, out):
    """20: Inclusive prefix sum of at most 1024 elements, in ONE block.

    Launch [1, threads] with len(x) <= threads <=1024; threads need not be a
    power of two. Shared storage is float64; output dtype controls rounding.
    Out-of-range threads must still participate in every barrier.
    """
    scratch = cuda.shared.array((2, 1024), float64)
    t = cuda.threadIdx.x
    scratch[0, t] = 0.0
    if t < x.size:
        scratch[0, t] = x[t]
    cuda.syncthreads()
    src = 0
    offset = 1
    while offset < cuda.blockDim.x:
        value = scratch[src, t]
        if t >= offset:
            value += scratch[src, t - offset]
        scratch[1 - src, t] = value
        cuda.syncthreads()
        src = 1 - src
        offset *= 2
    if t < x.size:
        out[t] = scratch[src, t]


@cuda.jit
def scan_blelloch_work_efficient(x, out):
    """21: Inclusive prefix sum in ONE power-of-two block of <=1024 threads.

    Choose threads = next_power_of_two(len(x)), including for non-power-of-two
    inputs. Padding is zero. Tree storage inserts one padding element per 32
    logical elements to reduce bank conflicts. The exclusive down-sweep result
    is converted to inclusive by adding the original x[t]. Distinct input/output.
    """
    scratch = cuda.shared.array(1056, float64)
    t = cuda.threadIdx.x
    padded_t = t + t // 32
    scratch[padded_t] = 0.0
    if t < x.size:
        scratch[padded_t] = x[t]
    cuda.syncthreads()
    stride = 1
    while stride < cuda.blockDim.x:
        index = (t + 1) * 2 * stride - 1
        if index < cuda.blockDim.x:
            left = index - stride
            scratch[index + index // 32] += scratch[left + left // 32]
        cuda.syncthreads()
        stride *= 2
    if t == 0:
        last = cuda.blockDim.x - 1
        scratch[last + last // 32] = 0.0
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride:
        index = (t + 1) * 2 * stride - 1
        if index < cuda.blockDim.x:
            left_index = index - stride
            left_index += left_index // 32
            index += index // 32
            left = scratch[left_index]
            scratch[left_index] = scratch[index]
            scratch[index] += left
        cuda.syncthreads()
        stride //= 2
    if t < x.size:
        out[t] = scratch[padded_t] + x[t]


@cuda.jit
def histogram_privatized(data, bins_out, num_bins):
    """22: Shared per-block histogram of integer bins in [0, num_bins).

    Require 1 <= num_bins <=256, int32 bins_out of length >= num_bins,
    initialized to zero (or existing counts to accumulate into). Values outside
    the range are ignored. Any positive grid/block sizes, block <=256.
    """
    private = cuda.shared.array(256, int32)
    t = cuda.threadIdx.x
    for b in range(t, num_bins, cuda.blockDim.x):
        private[b] = 0
    cuda.syncthreads()
    for i in range(cuda.grid(1), data.size, cuda.gridsize(1)):
        value = data[i]
        if 0 <= value < num_bins:
            cuda.atomic.add(private, value, 1)
    cuda.syncthreads()
    for b in range(t, num_bins, cuda.blockDim.x):
        cuda.atomic.add(bins_out, b, private[b])


@cuda.jit
def dynamic_work_queue(queue_in, queue_out, counter):
    """23: Atomically compact strictly positive (>0) input values.

    counter is a one-element int32 array initially zero; queue_out capacity
    must be >= len(queue_in). Only queue_out[:counter[0]] is defined. Order is
    intentionally unspecified. Grid-stride loop permits undersubscribed grids.
    """
    for i in range(cuda.grid(1), queue_in.size, cuda.gridsize(1)):
        value = queue_in[i]
        if value > 0:
            slot = cuda.atomic.add(counter, 0, 1)
            queue_out[slot] = value


@cuda.jit
def atomic_cas_float_add(arr, index, val):
    """24: Each launched thread adds val once to float32 storage via int CAS.

    Pass INT32 backing storage, e.g. cuda.to_device(float_array.view(np.int32)),
    and view the copied result as float32 on the host. index must be valid.
    Uses genuine libdevice.int_as_float/float_as_int and integer comparison, so
    NaN does not cause a nonterminating retry loop. Requires real CUDA; the
    simulator does not implement libdevice bitcasts. One-dimensional block<=256.
    """
    old = arr[index]
    while True:
        assumed = old
        value = libdevice.int_as_float(assumed)
        replacement = libdevice.float_as_int(value + val)
        old = cuda.atomic.cas(arr, index, assumed, replacement)
        if old == assumed:
            break


@cuda.jit
def sparse_dot_product_atomic(indices, values, dense_vec, result):
    """25: Accumulate COO sparse/dense dot into result[0] using atomic add.

    indices/values have equal lengths, indices valid for dense_vec; duplicate
    indices are supported. result is a one-element float32/float64 array,
    initialized to zero. Reduction order (and roundoff) is nondeterministic.
    """
    for i in range(cuda.grid(1), values.size, cuda.gridsize(1)):
        cuda.atomic.add(result, 0, values[i] * dense_vec[indices[i]])