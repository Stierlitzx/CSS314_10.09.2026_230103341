"""
Task 3: Arbitrary-Size Vector Scaling via Grid-Stride Loops
==========================================================
Kernel processes arbitrary vector sizes with a stride loop.

Functions:
    grid_stride_scale_kernel(...) -> @cuda.jit def grid_stride_scale_kernel(d_arr, factor, N)
    run_grid_stride(h_arr, factor)-> host caller returning the result array
"""

import numpy as np

try:
    from numba import cuda
    _CUDA_AVAILABLE = bool(getattr(cuda, "is_available", lambda: False)())
except Exception:  # pragma: no cover
    _CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# CUDA kernel -- exactly as specified in the assignment
# ---------------------------------------------------------------------------
if _CUDA_AVAILABLE:
    @cuda.jit
    def grid_stride_scale_kernel(d_arr, factor, N):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, N, stride):
            d_arr[i] = d_arr[i] * factor
else:  # pragma: no cover
    # CPU fallback -- simulate grid-stride with 256*64 = 16384 threads
    TOTAL_THREADS = 256 * 64

    def grid_stride_scale_kernel(d_arr, factor, N):
        # Each emulated thread i handles elements i, i+TOTAL_THREADS, i+2*TOTAL_THREADS, ...
        for start in range(TOTAL_THREADS):
            if start >= N:
                break
            i = start
            while i < N:
                d_arr[i] = d_arr[i] * factor
                i += TOTAL_THREADS


# ---------------------------------------------------------------------------
# Host caller
# ---------------------------------------------------------------------------
def run_grid_stride(h_arr, factor):
    """Scale every element of h_arr by `factor` and return the result."""
    h_arr = np.asarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]

    if _CUDA_AVAILABLE:
        d_arr = cuda.to_device(h_arr)
        threads_per_block = 256
        blocks_per_grid = 64
        total_threads = threads_per_block * blocks_per_grid
        grid_stride_scale_kernel[blocks_per_grid, threads_per_block](
            d_arr, np.float32(factor), np.int32(N)
        )
        cuda.synchronize()
        return d_arr.copy_to_host()
    else:
        # CPU fallback -- full grid-stride logic over the whole array
        out = h_arr.copy()
        grid_stride_scale_kernel(out, float(factor), N)
        return out


if __name__ == "__main__":
    # Assignment test: size = 100_000, factor = 4.25
    N = 100_000
    test_arr = np.ones(N, dtype=np.float32)
    factor = 4.25
    res = run_grid_stride(test_arr, factor)
    assert np.allclose(res, factor), "Task 3 elements not uniformly scaled"
    print(f"TASK 3 PASSED: N={N}, factor={factor}, res[0]={res[0]}")
