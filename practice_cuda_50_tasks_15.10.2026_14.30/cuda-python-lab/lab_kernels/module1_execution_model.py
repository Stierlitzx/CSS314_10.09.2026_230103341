"""Tasks 1–8: CUDA execution geometry and guarded elementwise kernels.

Launch with Numba's ``kernel[grid, block](...)`` syntax. Inputs and outputs
must have matching shapes (except transpose/RGB as documented), and launches
must cover the output domain. Empty arrays can use a one-block guarded launch.
No host/CPU fallback is provided. Unless explicitly safe elementwise, input and
output buffers must not overlap. Both float32 and float64 arrays are supported.
"""

from numba import cuda


@cuda.jit
def vector_add_1d(a, b, out):
    """Task 01: ``out[i] = a[i] + b[i]``; any 1D launch covering out.size."""
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


@cuda.jit
def vector_scale_grid_stride(x, alpha, out):
    """Task 02: scale x using any nonempty 1D grid, even an undersized one."""
    for i in range(cuda.grid(1), out.size, cuda.gridsize(1)):
        out[i] = alpha * x[i]


@cuda.jit
def matrix_add_2d(A, B, C):
    """Task 03: add equal-shaped matrices; launch axes (x, y) = (col, row)."""
    col, row = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


@cuda.jit
def matrix_transpose_naive(A, B):
    """Task 04: B=A.T, B.shape=A.shape[::-1]; 2D grid covers input (col,row)."""
    col, row = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        B[col, row] = A[row, col]


@cuda.jit
def volume_init_3d(vol, D, H, W):
    """Task 05: set vol[z,y,x]=z*10000+y*100+x for shape (D,H,W).

    Use a 3D grid with launch axes (x,y,z) covering (W,H,D), e.g. block
    (8,8,8). D,H,W are nonnegative and do not exceed the matching array axes.
    """
    x, y, z = cuda.grid(3)
    if x < W and y < H and z < D:
        vol[z, y, x] = z * 10000 + y * 100 + x


@cuda.jit
def rgb_to_grayscale(img_rgb, img_gray):
    """Task 06: BT.601 luma 0.299R+0.587G+0.114B, without normalization.

    RGB input is (height,width,3); output is a floating (height,width) array.
    Launch 2D with (x,y)=(col,row). uint8 or floating RGB inputs are accepted.
    """
    col, row = cuda.grid(2)
    if row < img_gray.shape[0] and col < img_gray.shape[1]:
        img_gray[row, col] = (0.299 * img_rgb[row, col, 0]
                              + 0.587 * img_rgb[row, col, 1]
                              + 0.114 * img_rgb[row, col, 2])


@cuda.jit
def vector_lerp(a, b, t, out):
    """Task 07: (1-t[i])*a[i]+t[i]*b[i]; 1D grid covers output.

    t is an equal-length vector of per-element interpolation weights; values
    outside [0,1] extrapolate without clipping.
    """
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (1.0 - t[i]) * a[i] + t[i] * b[i]


@cuda.jit
def vector_threshold(x, thresh, mask):
    """Task 08: write 1 iff x[i] >= thresh, otherwise 0; grid-stride 1D launch.

    Output is uint8 or bool. NaNs compare false, as in NumPy. Any nonempty
    grid covers the full input through a grid-stride loop.
    """
    for i in range(cuda.grid(1), mask.size, cuda.gridsize(1)):
        mask[i] = 1 if x[i] >= thresh else 0