"""
Task 2: 1D Boundary Stencil & Halo Protection
==============================================
3-point smoothing filter with boundary clamping (halo replication).

Functions:
    cpu_stencil(arr)       -> exact NumPy reference
    stencil_1d_kernel(...) -> @cuda.jit def stencil_1d(d_in, d_out, N) as specified
    run_stencil(h_in)      -> host wrapper (CUDA if available, else CPU fallback)
"""

import numpy as np

try:
    from numba import cuda
    _CUDA_AVAILABLE = bool(getattr(cuda, "is_available", lambda: False)())
except Exception:  # pragma: no cover
    _CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Exact CPU reference (must match the assignment)
# ---------------------------------------------------------------------------
def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


# ---------------------------------------------------------------------------
# CUDA kernel -- exactly as specified in the assignment
# ---------------------------------------------------------------------------
if _CUDA_AVAILABLE:
    @cuda.jit
    def stencil_1d(d_in, d_out, N):
        idx = cuda.grid(1)
        if idx < N:
            # ---- boundary guard ----
            if idx == 0:
                left = d_in[0]          # left halo replication
            elif idx > 0:
                left = d_in[idx - 1]
            else:
                left = d_in[0]

            if idx == N - 1:
                right = d_in[N - 1]     # right halo replication
            elif idx < N - 1:
                right = d_in[idx + 1]
            else:
                right = d_in[N - 1]

            d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right
else:  # pragma: no cover
    # Fallback: identical logic running on CPU (numba.cuda unavailable)
    def stencil_1d(d_in, d_out, N):
        for i in range(N):
            if i == 0:
                left = d_in[0]
            else:
                left = d_in[i - 1]

            if i == N - 1:
                right = d_in[N - 1]
            else:
                right = d_in[i + 1]

            d_out[i] = 0.25 * left + 0.5 * d_in[i] + 0.25 * right


# ---------------------------------------------------------------------------
# Host wrapper -- launches blocks of 256 threads
# ---------------------------------------------------------------------------
def run_stencil(h_in):
    """Run the 1D stencil on h_in and return the result as a numpy array."""
    h_in = np.asarray(h_in, dtype=np.float32)
    N = h_in.shape[0]

    if _CUDA_AVAILABLE:
        d_in = cuda.to_device(h_in)
        d_out = cuda.device_array_like(d_in)
        block_dim = 256
        grid_dim = (N + block_dim - 1) // block_dim
        stencil_1d[grid_dim, block_dim](d_in, d_out, np.int32(N))
        cuda.synchronize()
        return d_out.copy_to_host()
    else:
        # CPU fallback -- same math, no CUDA translation
        out = np.empty_like(h_in)
        stencil_1d(h_in, out, N)
        return out


if __name__ == "__main__":
    # --- quick self-test with the assignment's odd, non-power-of-two size ---
    N = 10007
    test_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)
    h_gpu = run_stencil(test_in)
    h_cpu = cpu_stencil(test_in)
    delta = np.max(np.abs(h_gpu - h_cpu))
    print(f"TASK 2: N={N}, max delta = {delta:.3e}")
    assert np.allclose(h_gpu, h_cpu, atol=1e-4), "Task 2 output mismatch"
    print("TASK 2 PASSED: MAX DELTA = %.3e" % delta)

