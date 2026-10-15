# Advanced GPU Computing with CUDA Python — Contract Summary

Source: the user-supplied text of *50 Comprehensive Practical Tasks | Zero-Intervention
Grading Suite for Google Colab & Git*, supplied after the original damaged PDF.
Target: Colab Tesla T4 / L4 / A100, Python 3.10+, Numba CUDA and CuPy.
This is a concise implementation checklist, not a claim of instructor certification.

## Module 1 — Execution Model (01–08)

| Task | Required signature | Contract |
|---|---|---|
| 01 | `vector_add_1d(a, b, out)` | Float32 addition, guard arbitrary N; 256 threads, ceil(N/256) blocks. |
| 02 | `vector_scale_grid_stride(x, alpha, out)` | Grid-stride scaling with a grid smaller than N. |
| 03 | `matrix_add_2d(A, B, C)` | Row/y, column/x indexing; 16×16 threads; rectangular tails. |
| 04 | `matrix_transpose_naive(A, B)` | Global-memory transpose into (N,M). |
| 05 | `volume_init_3d(vol, D, H, W)` | 8×8×8 threads; value z*10000+y*100+x. |
| 06 | `rgb_to_grayscale(img_rgb, img_gray)` | RGB (H,W,3) → float luminance, 0.299R+0.587G+0.114B. |
| 07 | `vector_lerp(a, b, t, out)` | Per-element interpolation (1-t)*a+t*b. |
| 08 | `vector_threshold(x, thresh, mask)` | Grid-stride uint8/bool mask for x>=threshold. |

## Module 2 — Memory Hierarchy (09–17)

| Task | Required signature | Contract |
|---|---|---|
| 09 | `benchmark_strided_access(in_arr, out_arr, stride)` | Correct strided read; benchmark strides 1,2,4,8,16,32. |
| 10 | `stencil_1d_shared(in_arr, out_arr, radius)` | Shared tile plus clamped halos; nearest-boundary moving average. |
| 11 | `conv_1d_shared_tiled(x, kernel, y)` | Shared tile/filter; zero-padded convolution, NumPy same mode. |
| 12 | `matmul_tiled_shared(A, B, C)` | 16×16 shared A/B tiles; rectangular edges. |
| 13 | `transpose_bank_conflict_free(A, B)` | 32×33 padded tile; coalesced transpose. |
| 14 | `reduction_sum_shared(in_arr, block_sums)` | Per-block sequential-addressed tree sum. |
| 15 | `reduction_argmax_shared(in_arr, block_max_vals, block_max_indices)` | Per-block max plus int64 index; earliest index wins ties. |
| 16 | `constant_memory_filter(signal, out)` | Fixed 64-tap FIR in cuda.const.array_like, same convolution. |
| 17 | `benchmark_pinned_transfer(size_bytes)` | Both transfer directions, integrity and event bandwidth, pinned vs pageable. |

## Module 3 — Warp Intrinsics & Atomics (18–25)

| Task | Required signature | Contract |
|---|---|---|
| 18 | `warp_reduce_sum_shfl(in_arr, out_arr)` | Shuffle reduction deltas16,8,4,2,1; aggregate to one sum per block. |
| 19 | `warp_vote_analyzer(data, mask_out)` | all/any/ballot for data>0, one predicate bitmask per warp. |
| 20 | `scan_hillis_steele(x, out)` | Shared double-buffered single-block inclusive scan, <=1024 items. |
| 21 | `scan_blelloch_work_efficient(x, out)` | Padded up/down-sweep tree; exclusive converted to inclusive. |
| 22 | `histogram_privatized(data, bins_out, num_bins)` | Private shared bins, atomic merge to initially zero global bins. |
| 23 | `dynamic_work_queue(queue_in, queue_out, counter)` | Atomic compaction; output capacity >=input length, counter initialized zero. |
| 24 | `atomic_cas_float_add(arr, index, val)` | Int32 CAS loop over float32 bits, concurrent updates. |
| 25 | `sparse_dot_product_atomic(indices, values, dense_vec, result)` | Grid-stride NNZ products, atomic sum to initially zero result. |

## Module 4 — Scientific Computing (26–34)

| Task | Required signature | Contract |
|---|---|---|
| 26 | `heat_diffusion_2d_step(u_curr, u_next, alpha, dt, dx, dy)` | Jacobi update with x/y spacing and fixed boundary edges. |
| 27 | `wave_equation_1d_step(u_prev, u_curr, u_next, c, dt, dx)` | Shared radius1 tile, fixed edges, C=c*dt/dx, stable CFL. |
| 28 | `nbody_all_pairs(positions, velocities, masses, dt, softening)` | Shared source-body tiles; update velocities only, G=1, epsilon squared. |
| 29 | `gaussian_blur_2d_separable(img_in, img_out, temp_buf, kernel_1d)` | Shared horizontal/vertical passes, Gaussian reference sigma2. |
| 30 | `sobel_edge_detector(img_in, magnitude, orientation)` | Shared 3×3 apron, clamped edges, sqrt(Gx²+Gy²), atan2(Gy,Gx). |
| 31 | `monte_carlo_pi_cupy(num_samples)` | Batched GPU random points and inside-circle reduction. |
| 32 | `kmeans_assign_clusters(points, centroids, labels, distances)` | Nearest squared Euclidean distance; earliest centroid ties. |
| 33 | `fft_radix2_butterfly_stage(data_re, data_im, stage, N)` | One-based radix2 DIT stage with negative-exponent twiddles. |
| 34 | `mandelbrot_escape_time(out, xmin, xmax, ymin, ymax, max_iter)` | 2D complex grid, iterate z²+c until escape or max_iter. |

## Module 5 — Streams & CuPy (35–42)

| Task | Required signature | Contract |
|---|---|---|
| 35 | `pipeline_dual_stream(h_in, h_out, chunk_size)` | Pinned staging and two streams; overlap copies/compute when hardware permits. |
| 36 | `multistream_gemv(matrix, vectors, out_vectors, num_streams)` | Batched GEMV using round-robin streams and async staging. |
| 37 | `cupy_rawkernel_sinc(x, out)` | Raw CUDA normalized sin(pi*x)/(pi*x), 1 at zero. |
| 38 | `cupy_rawmodule_arithmetic(a, b, res_add, res_sub, res_mul)` | Three extern-C arithmetic kernels in one RawModule. |
| 39 | `cuda_event_profiler(kernel_func, args, num_warmup, num_repeats)` | Warmup, event-based milliseconds, measured variance. |
| 40 | `cuda_graph_capture_exec(a, b, c, num_iterations)` | Capture multiple launches; repeat executable graph, validate procedural reference. |
| 41 | `benchmark_unified_memory(size_bytes)` | Managed allocation, CPU/GPU consistency, first-touch/prefetch timings if supported. |
| 42 | `grid_cooperative_barrier(arr, iterations)` | Cooperative launcher and grid-wide sync; occupancy-safe grid. |

## Module 6 — Deep Learning (43–50)

| Task | Required signature | Contract |
|---|---|---|
| 43 | `stable_online_softmax(logits, probs)` | Stable row normalization, shared max/sum reductions. |
| 44 | `fused_bias_gelu(x, bias, out)` | Bias modulo D and tanh-approximate GELU in one kernel. |
| 45 | `maxpool2d_with_argmax(x, out, argmax_idx, pool_h, pool_w, stride)` | NCHW valid pooling and flattened spatial argmax. |
| 46 | `batch_norm_1d_forward(x, gamma, beta, eps, out, mean_out, var_out)` | Featurewise population variance across B; affine normalization. |
| 47 | `layer_norm_welford(x, gamma, beta, eps, out)` | Per-row parallel Welford tuples; population variance. |
| 48 | `spmv_csr_kernel(row_ptr, col_indices, values, x, y, num_rows)` | CSR sparse matrix-vector multiplication. |
| 49 | `attention_sdpa_tiled(Q, K, V, Out, scale)` | Shared K/V tiles and online max/sum; softmax(scale*QKᵀ)V. |
| 50 | `fused_rope_embedding(q, cos_table, sin_table, q_rot)` | Adjacent-element rotary pairs and shared trigonometric tables. |

## Explicit conventions where the sheet is incomplete

- Queue predicate: values strictly greater than zero; order is nondeterministic.
- Task35 operation: doubling the input. The original sheet does not specify one.
- GEMV batches: columns of `vectors` (N,B); output (M,B), consistent with `matrix @ vectors`.
- Fixed FIR coefficients are exported by Module2 and used by its reference tests.
- CUDA constant memory uses `cuda.const.array_like` inside the kernel, not a decorator.
- Gaussian edges use SciPy's default reflect mode; Sobel uses clamped/nearest edges.
- FFT DIT requires bit-reversed input before stage1; the kernel handles one stage only.
- CAS task accepts an int32 view of float32 storage because CAS operates on integer bits.
- Finite floats and valid dimensions/layouts are expected unless a function documents otherwise.
- Fixed shared/local tile bounds are described in module docstrings. They are not
  claims of unbounded shape support.

## Validation and submission boundary

Functional assertions compare against independent NumPy/SciPy/CuPy references.
Large task-sheet workloads belong on real hardware, not the CPU simulator. Hardware
profiling requirements (bank conflicts, timeline overlap, prescribed speedup ratios)
remain external acceptance checks; no synthetic performance values are substituted.

The text specifies Git Classroom submission and instructor HMAC grading but supplies
neither the official tests nor a trusted signing service. The local harness and
token utility must not be represented as that missing infrastructure.