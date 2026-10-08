"""
Task 4: 2D Spatial Convolution - Sobel Horizontal Filter
=======================================================
2D grid geometry + boundary-safe 2D indexing.

Functions:
    sobel_x_kernel(...)     -> @cuda.jit def sobel_x_kernel(d_in, d_out, rows, cols)
    run_sobel(h_img)        -> host wrapper returning the filtered matrix
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
    def sobel_x_kernel(d_in, d_out, rows, cols):
        col, row = cuda.grid(2)
        if col >= 1 and col < cols - 1 and row >= 1 and row < rows - 1:
            # Internal pixels: Sobel-X derivative along columns
            d_out[col, row] = (
                -1 * d_in[col - 1, row - 1]
                + 1 * d_in[col + 1, row - 1]
                - 2 * d_in[col - 1, row]
                + 2 * d_in[col + 1, row]
                - 1 * d_in[col - 1, row + 1]
                + 1 * d_in[col + 1, row + 1]
            )
        else:
            # Outer border pixels: set to 0
            d_out[col, row] = 0.0
else:  # pragma: no cover
    # CPU fallback -- same indexing logic
    def sobel_x_kernel(d_in, d_out, rows, cols):
        for row in range(rows):
            for col in range(cols):
                if 1 <= col < cols - 1 and 1 <= row < rows - 1:
                    d_out[col, row] = (
                        -1 * d_in[col - 1, row - 1]
                        + 1 * d_in[col + 1, row - 1]
                        - 2 * d_in[col - 1, row]
                        + 2 * d_in[col + 1, row]
                        - 1 * d_in[col - 1, row + 1]
                        + 1 * d_in[col + 1, row + 1]
                    )
                else:
                    d_out[col, row] = 0.0


# ---------------------------------------------------------------------------
# Host wrapper -- launches 16x16 blocks, dynamic grid
# ---------------------------------------------------------------------------
def run_sobel(h_img):
    """Apply Sobel-X convolution and return the filtered matrix."""
    h_img = np.asarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape

    if _CUDA_AVAILABLE:
        d_in = cuda.to_device(h_img)
        d_out = cuda.device_array_like(d_in)
        threads_2d = (16, 16)
        blocks_x = (cols + threads_2d[0] - 1) // threads_2d[0]
        blocks_y = (rows + threads_2d[1] - 1) // threads_2d[1]
        sobel_x_kernel[blocks_x, blocks_y, threads_2d[0], threads_2d[1]](
            d_in, d_out, np.int32(rows), np.int32(cols)
        )
        cuda.synchronize()
        return d_out.copy_to_host()
    else:
        # CPU fallback -- same indexing logic
        out = np.zeros_like(h_img)
        sobel_x_kernel(h_img, out, rows, cols)
        return out


if __name__ == "__main__":
    # Assignment test: 64x64 ones
    test_img = np.ones((64, 64), dtype=np.float32)
    sobel_res = run_sobel(test_img)
    assert np.max(np.abs(sobel_res[1:-1, 1:-1])) < 1e-5, "Task 4 interior gradient != 0 for flat field"
    assert np.all(sobel_res[0, :] == 0.0), "Task 4 border rows not zeroed"
    assert np.all(sobel_res[:, 0] == 0.0), "Task 4 border columns not zeroed"
    print("TASK 4 PASSED: flat-field interior gradient == 0, borders zeroed")
