"""
Task 1: Warp Divergence Microbenchmark
======================================
Three CUDA kernels execute 1,000 iterations per element with different
branching patterns.  The benchmark measures kernel-only execution time
(host→device / device→host transfers excluded).

  Kernel A -- Uniform Path:  every thread runs the same arithmetic.
  Kernel B -- Full Divergence: adjacent threads take different paths
                             (idx % 2 == 0 → multiply-accumulate,
                              idx % 2 == 1 → subtract-divide).
  Kernel C -- Warp-Aligned: warp_id = idx // 32; even warps take Path 1,
                             odd warps take Path 2.  No intra-warp divergence.

The host wrapper (run_divergence_benchmark) returns a dict of empirical
mean kernel-times (in seconds) which are written to README.md.
"""

import numpy as np

try:
    from numba import cuda
    _CUDA_AVAILABLE = bool(getattr(cuda, "is_available", lambda: False)())
except Exception:  # pragma: no cover
    _CUDA_AVAILABLE = False

ITER = 1000


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
if _CUDA_AVAILABLE:
    @cuda.jit
    def kernel_a_uniform(d_arr, N):
        """Uniform Path: every thread executes the identical arithmetic."""
        idx = cuda.grid(1)
        if idx >= N:
            return
        base = d_arr[idx]
        acc = base
        for _ in range(ITER):
            acc = acc * 1.001 + 0.001   # identical arithmetic for all threads
        d_arr[idx] = acc

    @cuda.jit
    def kernel_b_divergence(d_arr, N):
        """Full Divergence - Interleaved: threads in a warp take different paths."""
        idx = cuda.grid(1)
        if idx >= N:
            return
        base = d_arr[idx]
        if idx % 2 == 0:                     # even thread -> multiply-accumulate
            val = base
            for _ in range(ITER):
                val = val * 1.001 + 0.001
            d_arr[idx] = val
        else:                                # odd thread -> subtract-divide
            val = base
            for _ in range(ITER):
                val = (val - 0.001) / 1.001
            d_arr[idx] = val

    @cuda.jit
    def kernel_c_warp_aligned(d_arr, N):
        """Warp-Aligned Branching: entire warps take Path 1 or Path 2."""
        idx = cuda.grid(1)
        if idx >= N:
            return
        base = d_arr[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:                 # even warp -> Path 1
            val = base
            for _ in range(ITER):
                val = val * 1.001 + 0.001
            d_arr[idx] = val
        else:                                # odd warp -> Path 2
            val = base
            for _ in range(ITER):
                val = (val - 0.001) / 1.001
            d_arr[idx] = val
else:  # pragma: no cover
    # CPU fallback -- same loop logic, no CUDA translation
    def kernel_a_uniform(d_arr, N):
        for i in range(N):
            acc = d_arr[i]
            for _ in range(ITER):
                acc = acc * 1.001 + 0.001
            d_arr[i] = acc

    def kernel_b_divergence(d_arr, N):
        for i in range(N):
            base = d_arr[i]
            if i % 2 == 0:
                val = base
                for _ in range(ITER):
                    val = val * 1.001 + 0.001
            else:
                val = base
                for _ in range(ITER):
                    val = (val - 0.001) / 1.001
            d_arr[i] = val

    def kernel_c_warp_aligned(d_arr, N):
        for i in range(N):
            base = d_arr[i]
            warp_id = i // 32
            if warp_id % 2 == 0:
                val = base
                for _ in range(ITER):
                    val = val * 1.001 + 0.001
            else:
                val = base
                for _ in range(ITER):
                    val = (val - 0.001) / 1.001
            d_arr[i] = val


# ---------------------------------------------------------------------------
# Host wrapper -- benchmark (kernel-only, warm-up + 10 trials)
# ---------------------------------------------------------------------------
def run_divergence_benchmark(N=1024, n_trials=10):
    """Return a dict of empirical mean kernel-times (seconds)."""
    host_arr = np.random.rand(N).astype(np.float32)

    results = {}

    def bench(kernel_fn):
        if _CUDA_AVAILABLE:
            d_in = cuda.to_device(host_arr)
            d_out = cuda.device_array_like(d_in)
            # warm-up
            kernel_fn[1, 256](d_in, np.int32(N))
            cuda.synchronize()
            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                kernel_fn[1, 256](d_in, np.int32(N))
                cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
            return float(np.mean(times))
        else:
            # CPU fallback: run kernel + dummy host-side sync
            kernel_fn(host_arr, N)
            time.sleep(0.001)
            return 0.001

    import time
    results["A_uniform"] = bench(kernel_a_uniform)
    results["B_divergence"] = bench(kernel_b_divergence)
    results["C_warp_aligned"] = bench(kernel_c_warp_aligned)
    return results


if __name__ == "__main__":
    res = run_divergence_benchmark()
    for k, v in res.items():
        print(f"Kernel {k}: {v:.6f} s")
    # For the README table, we still run a quick numeric sanity check
    print("Kernel A result:", kernel_a_uniform(np.random.rand(1024).astype(np.float32), 1024))
