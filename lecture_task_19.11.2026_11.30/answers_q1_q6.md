### Analytical & Forensic Engineering Responses

All numbers below come from the Colab run shown above: **Tesla T4 (CC 7.5), UUID `GPU-198f4157-3506-5952-ab80-80b6dd077a59`,
`INSTANCE_SEED = 15753744789197117896 (0xdaa08b6b3a4159c8)`, Open MPI 4.1.6, 4 ranks in a 2 × 2 grid.**

---

- **Question 1 (Memory Coalescing & Strided Column Packing):**
  The padded array is `float64[258, 258]` in C order, so one row is 258 × 8 = **2,064 bytes** and element `u[i, j]` lives at
  byte `8·(258·i + j)`. The top/bottom strips `u[1, 1:257]` and `u[256, 1:257]` are 2,048 contiguous bytes, but the left/right
  strips `u[1:257, 1]` and `u[1:257, 256]` are 256 doubles spaced 2,064 bytes apart. A copy engine (`cudaMemcpyAsync`) moves one
  contiguous byte range, so it cannot fetch only those 256 values in one 1D DMA; it would have to copy the whole span between them.
  Inside `pack_halos_kernel`, a warp of 32 threads reading the row strip touches 32 consecutive doubles = 256 B = 8 32-byte sectors
  (100 % sector efficiency). The same warp reading a column touches 32 addresses 2,064 B apart, i.e. **32 separate sectors that each
  carry only 8 useful bytes (25 % efficiency)**. That uncoalesced read only happens once, on the GPU, in L2/DRAM. The writes into
  `send_buf[2..3, k]` are fully coalesced, and PCIe then carries one contiguous 8 KiB block (`h_send`, 4 × 256 × 8 B).
  **Cost of an unstrided bounding-box copy instead:** the smallest contiguous span holding column 1 runs from flat index 259
  (`u[1,1]`) to 66,049 (`u[256,1]`), which is **65,791 doubles = 526,328 B for 2,048 useful bytes, so 524,280 extra bytes per column**.
  Both columns add **1,048,560 B per rank per step** (≈ 87 µs at a sustained 12 GB/s Gen3 ×16, on top of the measured 209.56 µs/step
  for the whole Stage A). Across 4 ranks × 400 steps that is **1,677,696,000 extra bytes (≈ 1.68 GB) device→host**, and the same again
  host→device if the ghost columns were injected the same way. Our packed path moves **13,107,200 B** D2H in total for the whole run.

- **Question 2 (Shared-Memory Tile Indexing & Bank Conflict Analysis):**
  In the kernel `sy = ty + 1`, `sx = tx + 1` and `smem` is `float64[18, 18]`, so `smem[sy, sx]` sits at byte `8·(18·sy + sx)`.
  For a full interior tile:

  | thread | writes (before `bar.sync`) | reads (after `bar.sync`) |
  |---|---|---|
  | `(tx=0, ty=0)` | `smem[1,1]` ← `u[i,j]`, `smem[0,1]` ← `u[i-1,j]` (top apron), `smem[1,0]` ← `u[i,j-1]` (left apron) | `smem[1,1]`, `smem[0,1]`, `smem[2,1]`, `smem[1,0]`, `smem[1,2]` |
  | `(tx=15, ty=15)` | `smem[16,16]` ← `u[i,j]`, `smem[17,16]` ← `u[i+1,j]` (bottom apron), `smem[16,17]` ← `u[i,j+1]` (right apron) | `smem[16,16]`, `smem[15,16]`, `smem[17,16]`, `smem[16,15]`, `smem[16,17]` |

  The four corners `smem[0,0]`, `smem[0,17]`, `smem[17,0]`, `smem[17,17]` are never touched (5-point stencil).
  Bank mapping on CC 7.5: 32 banks × 4 B, and a `float64` covers two adjacent banks, `2·(18·sy+sx) mod 32` and `+1`.
  A 16 × 16 block puts rows `ty = 2w, 2w+1` in warp `w`. A 64-bit warp access (256 B) is always served as two half-warp passes of
  128 B, which is the shared-memory width per cycle and not a conflict. **Row-wise accesses** (`smem[sy, sx]`, `smem[sy±1, sx]`, and
  the shifted `smem[sy, sx±1]`) give each half-warp 16 consecutive doubles = 128 contiguous bytes that hit all 32 banks exactly once,
  so they are **conflict-free**. **Column-wise accesses** would conflict: consecutive rows are 18 doubles = 36 words apart, so the
  bank of row k is `4k mod 32`. Rows k and k+8 then land on the same bank pair, and a half-warp reading 16 rows of one column has a
  **2-way bank conflict**. Our kernel never does this; its only column-direction accesses are the `tx==0` / `tx==15` apron stores,
  2 threads per warp at bank offset 4, so no conflict. An odd tile width (17 or 19 doubles, row pitch 34 or 38 words) would make
  column access conflict-free as well.

- **Question 3 (PTX Assembly Audit & Warp Synchronization):**
  `tiled_stencil_kernel.inspect_asm()` in this run contains **`st.shared.f64` = 5, `ld.shared.f64` = 5, `bar.sync` = 1**
  (`barrier.sync` = 0), so `ptx_audit_passed = true`. The counts map one-to-one onto the source: 5 guarded stores (centre + 4 apron
  edges), one `bar.sync 0` from `cuda.syncthreads()`, and 5 loads (centre + N/S/W/E). The offsets confirm the layout:
  `st.shared.f64 [%rd106+136]` is +17 doubles (right apron column), `ld.shared.f64 [%rd156+288]` is +36 doubles = two 18-wide rows
  (south neighbour relative to north), and `[%rd167+16]` is +2 doubles (east relative to west).
  **Hazard without the barrier:** this is a cross-warp read-after-write race. Warp `w` owns tile rows `ty = 2w, 2w+1`, but its stencil reads
  `smem[sy-1, sx]` and `smem[sy+1, sx]`, which belong to the rows of warps `w-1` and `w+1`. Apron row 0 is written only by warp 0
  (`ty==0`) and apron row 17 only by warp 7 (`ty==15`). The 8 warps are scheduled independently, so without `bar.sync` warp 3 can issue
  its `ld.shared` before warps 2 and 4 have executed their `st.shared`. It then reads stale data left in that shared-memory region by a
  previous block. The update is silently wrong and changes between runs, so the parity check (here exactly **0.0**) would fail. One
  barrier is enough because the tile is written once and only read afterwards (no write-after-read reuse).

- **Question 4 (Multi-Process Context Sharing on a Single Physical GPU):**
  `mpirun -np 4` creates 4 OS processes (PIDs 969–972 in the topology table), each with its own CUDA context on the one T4.
  Without MPS, contexts from different processes do not run kernels concurrently. The GPU time-slices between them, and each
  switch saves/restores context state. Inside one context, our two streams can overlap, and the T4's copy engines can run DMA
  in parallel with kernels of the active context. All 4 ranks share those 40 SMs, the 2 copy engines and one PCIe link.
  **Measured:** `serialized_ms = 3869.639 ms` (9.674 ms/step) vs `overlapped_ms = 1627.154 ms` (4.068 ms/step), a **2.378×
  speed-up**, with parity diff **0.0**. The CUDA events show that compute is tiny: the 254 × 254 interior kernel costs **298.33 µs/step**
  and Stage A (pack + D2H + sync) **209.56 µs/step**. The serialized step spans **9,338 µs** between its events, so over 90 % of a
  serialized step is waiting: host synchronisation, MPI hand-shakes with the slowest neighbour, and context time-slices. Stage C
  in overlapped mode (Waitall + H2D + unpack + 4 ring launches) still accounts for **3,191 µs/step**, and that is mostly the MPI wait. The
  overlapped schedule wins mainly because each rank enqueues its interior kernel before blocking in `Waitall`. The GPU then works
  through the ranks' interior kernels while the processes wait for each other, and only the thin boundary ring stays on the
  critical path after communication. On one shared GPU the gain therefore comes from hiding latency and scheduling gaps, not
  from extra hardware, and it is capped by the fact that all four ranks' kernels still go through the same SMs one context at a time.
  **With four dedicated GPUs over InfiniBand,** each rank has its own SMs and copy engines and no time-slicing. Ideal overlap then
  hides the network term behind the interior compute, so the step time approaches `max(T_interior, T_comm) + T_ring`. The gain is bounded by
  `T_comm / (T_interior + T_comm)` and grows with subdomain size. With GPUDirect RDMA the pinned host staging disappears completely.

- **Question 5 (Point-to-Point Message Deadlock & `MPI.PROC_NULL` Semantics):**
  A blocking `Send` returns as soon as the user buffer can be reused. **Below the transport's eager limit** (a few KiB for Open MPI's
  shared-memory BTL; our strips are 256 × 8 = **2,048 B**), the library copies the message into an internal buffer and `Send` returns
  at once, so the naive code happens to work on small grids. **Above the eager limit** it switches to the **rendezvous** protocol:
  the sender sends a request-to-send and blocks until the receiver posts the matching `Recv` and replies clear-to-send. If every rank
  executes the same fixed order, e.g. `Send(up); Send(down); Send(left); Send(right); Recv(...)`, then rank 0 `(0,0)` blocks in
  `Send(down → 2)` waiting for rank 2 to post `Recv(up)`. Rank 2 is blocked in its own `Send(up → 0)` waiting for rank 0 to post
  `Recv(down)`. Neither reaches its `Recv`, and the same holds for the 0↔1 and 2↔3 pairs. That is a circular wait: a deadlock that
  appears only when the grid is large enough (e.g. > 512 columns at a 4 KiB limit) to push the halos over the threshold. Our solver
  avoids this by posting all 4 `Irecv` before the 4 `Isend` and completing them with one `MPI.Request.Waitall`, so no call blocks before
  all requests are posted. (`Sendrecv` or odd/even ordering would also work.)
  **`MPI.PROC_NULL`:** `Create_cart(periods=[False, False])` makes `Shift` return `PROC_NULL` beyond the global edge. Rank 0 at
  `(0,0)` gets `up = left = PROC_NULL` and `down = 2`, `right = 1`, exactly as printed in the Task 1 table. A send to or receive from
  `PROC_NULL` completes immediately, transfers nothing and leaves the buffer untouched. So every rank, corners included, runs the
  **same 8 calls with no `if` on its position**, and two of rank 0's eight requests are free no-ops. The unpack kernel receives
  `has_up/has_left = 0` for those edges, so the ghost row/column keeps its initial **u = 0**, which is the homogeneous Dirichlet condition.

- **Question 6 (Explicit CFL Stability & Energy Conservation Proof):**
  From the run: `kappa_max = 0.2399984006` (Allreduce MAX; range `[0.080001, 0.239998]`), `Δx = Δy = 1/513 = 1.9493177388e-3`,
  `Δx² = 3.7998396e-6`.
  `Δt_max = Δx² / (4·κ_max) = 3.7998396e-6 / 0.9599936 = **3.958193e-6**`. The run used `Δt = 0.9·Δt_max = **3.562373e-6**`, so the
  diffusion number is `κ_max·Δt/Δx² = 0.225 < 0.25`. The interface conductivities are arithmetic means of κ, so none exceeds κ_max
  and the bound is conservative. The reaction term adds `3λu²·Δt ≤ 3·0.15·1.3204²·3.56e-6 = 2.8e-6`, which is negligible.
  400 steps simulate `t = 1.424949e-3`.

  | step | E = Σ u Δx Δy (Allreduce) | ΔE over previous 100 steps | L2 residual ‖uⁿ−uⁿ⁻¹‖₂ | max\|u\| |
  |---:|---:|---:|---:|---:|
  | 0 | 7.606762073269764e-02 | — | — | 1.3203510596 |
  | 100 | 7.606569852616432e-02 | −1.922207e-06 | 2.426829e-02 | 1.2822818371 |
  | 200 | 7.606388322666909e-02 | −1.815299e-06 | 2.284185e-02 | 1.2462127205 |
  | 300 | 7.606216093689147e-02 | −1.722290e-06 | 2.159952e-02 | 1.2121959096 |
  | 400 | 7.606052255966575e-02 | −1.638377e-06 | 2.050234e-02 | 1.1798519187 |

  E decreases strictly at every checkpoint (total −7.098e-6, relative −9.33e-5), and the step residual and max|u| also fall
  monotonically. These are signs of a stable, dissipative scheme; an unstable explicit step would make the residual grow
  geometrically. **Energy budget:** summing the update over all cells, the flux-form diffusion terms cancel pairwise between
  neighbours (the scheme is conservative), so `dE/dt = (boundary flux) − λ Σ u³ ΔxΔy + Σ S ΔxΔy`. The seeded source is a
  zero-net dipole (printed `∫S dA = 0.000e+00`). The pulses sit ≥ 6σ from the walls, so the boundary flux is ~1e-8. That leaves
  `dE/dt ≈ −λ∫u³ < 0`, which explains the monotone decay. For the three seeded Gaussians, `∫u³ = Σ A³·2πσ²/3 = 0.036602`, which
  predicts `ΔE ≈ −0.15 · 0.036602 · (100·Δt) = −1.956e-6` over steps 0→100. That agrees with the measured **−1.922e-6** to 1.7 %.
  The gradual drop in |ΔE| (1.92 → 1.64 e-6) follows the cubic damping as the peak amplitude decays from 1.32 to 1.18.
  Serialized and overlapped traces are identical to all 16 printed digits.
