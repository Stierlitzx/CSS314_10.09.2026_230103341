### Analytical & Forensic Engineering Responses

Run cited below (outputs above, manifest `hybrid_lab_manifest.json`, sha256 `22552c3d31565d6c…`, `verification_passed: true`):
**Tesla T4 (CC 7.5) `GPU-6c19b838-7a5d-1355-0fdd-83ef8e21deb2`**. Host: **Intel Xeon @ 2.00 GHz, 2 vCPUs** (`lscpu`: one
L1d instance of 32 KiB, L2 1 MiB, L3 38.5 MiB), 64-byte lines, Open MPI 4.1.6. `INSTANCE_SEED = 8070547215371552121`
(`0x700057eecb3d4579`) = SHA-256(`Stierlitzx::GPU-6c19…::x86_64::2`). All 4 ranks report CPU affinity `[0, 1]`.
For contrast I also ran the identical script locally on a 6-core / 12-thread Ryzen 5 7640HS with MS-MPI
(`local_ryzen_shm_crosscheck.txt`). Only its Tier-4 numbers are meaningful, because its GPU tiers ran in the CUDA simulator.

---

- **Question 1 (GPU Coalescing vs. CPU Cache-Line Granularity):**
  `u` is `float64[258, 258]` row-major, so consecutive rows are **2,064 B** apart, and the column `u[1:257, 1]` is 256
  doubles each 2,064 B from the next. **On the host CPU** the unit of transfer and coherence is a **64-byte line**.
  Each strided element drags a whole line (8 useful bytes of 64 = **12.5 %** efficiency) into L1, and every element
  lives in a different line. The adjacent-line prefetcher may fetch the buddy line too, which is another 64 B of waste.
  **On the T4** the L2/DRAM unit is a **32-byte sector** (4 per 128-byte L1 line). A warp of 32 threads reading 32 column
  elements issues 32 different sector requests, 32 × 32 B = 1,024 B for 256 useful bytes (**25 %** efficiency). The same
  warp reading a row strip needs 8 sectors (**100 %**). The finer sector makes a strided GPU read waste half as much
  as the CPU's 64-byte line. More importantly, `pack_halos_kernel` confines that waste to on-device DRAM traffic
  (≈ 8 KiB of sectors per column). It then writes all four strips **coalesced** into a contiguous `(4, 256)` buffer, so
  PCIe carries exactly **4 × 256 × 8 = 8,192 B per rank per step** (measured Stage A, pack + D2H + sync:
  **358.42 µs/step**). **Avoided overhead:** without packing, a contiguous DMA covering column 1 must span flat
  indices 259 → 66,049, i.e. **65,791 doubles = 526,328 B for 2,048 useful B (+524,280 B)**, and both columns add
  **1,048,560 B per rank per step**. Over 4 ranks × 300 steps that is **1,258,272,000 B ≈ 1.26 GB** of extra D2H
  (and as much again H2D if the ghost columns were written the same way), against **9,830,400 B** actually staged.

- **Question 2 (Shared-Memory Tile Synchronization & PTX Audit):**
  `tiled_stencil_kernel.inspect_asm()` in this run contains **`st.shared.f64` = 5, `ld.shared.f64` = 5, `bar.sync` = 1**
  (`barrier.sync` = 0, `ptx_audit_passed = true`): five guarded stores (centre + top/bottom/left/right apron), one
  `bar.sync 0` from `cuda.syncthreads()`, and five loads (centre + N/S/W/E). The offsets confirm the 18-wide `float64`
  tile: `st.shared.f64 [%rd106+136]` is +17 doubles (right apron), `ld.shared.f64 [%rd156+288]` is +36 doubles = two rows
  (south vs north neighbour), and `[%rd167+16]` is +2 doubles (east vs west). Thread `(0,0)` writes `smem[1,1]`,
  `smem[0,1]`, `smem[1,0]`; thread `(15,15)` writes `smem[16,16]`, `smem[17,16]`, `smem[16,17]`.
  **RAW hazard without the barrier:** a 16 × 16 block has 8 warps, and warp *w* holds rows `ty = 2w, 2w+1`. Each
  thread's stencil reads `smem[sy±1, sx]`, which for the warp's edge rows were written by warps *w−1* and *w+1*. Apron
  row 0 is written only by warp 0 and apron row 17 only by warp 7. Warps are scheduled independently, so without
  `bar.sync` a warp can execute its `ld.shared` before its neighbour's `st.shared`. It then reads stale shared memory
  left by a previous block: a read-after-write race whose result depends on scheduling. The update would be wrong,
  vary between runs, and fail the serialized/overlapped parity check (here exactly **0.0**).

- **Question 3 (Dual-Stream Compute-Communication Overlap on Shared GPU Silicon):**
  Measured over 300 steps (slowest rank): **`gpu_serialized_ms = 1182.134`** (3.94 ms/step) and **`gpu_overlapped_ms =
  1411.503`** (4.71 ms/step). On this VM the overlapped pipeline was **0.838×**, i.e. *slower*, with bit-identical
  results. The CUDA events explain why. The GPU work that overlap can hide is tiny: the 254 × 254 interior kernel takes
  **315.29 µs/step**. The step is dominated by host-side waiting: the serialized step spans **3,720.81 µs** between its
  events, and the overlapped Stage C (`Waitall` + H2D + unpack + 4 ring kernels) **3,636.83 µs/step**. The decisive
  factor is the host: **4 MPI ranks share 2 vCPUs** (every rank's affinity is `[0, 1]`). Each step needs CPU time per
  rank for kernel launches, a `stream.synchronize()` on the pinned send buffers, MPI posting, and `Waitall`, which Open
  MPI implements by busy-polling and so burns the time slices other ranks need. The overlapped path adds 4 ring launches
  and 6 event records per step. On an oversubscribed host, those extra host operations cost more than the ≈ 0.3 ms of
  interior compute they hide. (The morning lab's solver, same pipeline on a different Colab T4 VM, measured 2.38×. On
  this platform the result is governed by host scheduling, not by the GPU.)
  **Shared GPU vs dedicated GPUs:** the 4 processes create 4 CUDA contexts on one T4. Without MPS the GPU time-slices
  between contexts, so kernels from different ranks never run concurrently. All D2H/H2D copies from all ranks queue on
  the same copy engines and the same PCIe ×16 link, and each copy still bounces through each rank's pinned host buffer
  and the host MPI library. With **four dedicated GPUs and GPUDirect RDMA**, each rank has its own SMs, copy engines and
  PCIe path, and the NIC reads and writes GPU memory peer-to-peer. The D2H → MPI → H2D staging disappears, no host CPU
  sits in the data path, and the interior kernel can genuinely hide the network latency, so the gain approaches
  `min(T_comm, T_interior)` per step.

- **Question 4 (MPI-3 Shared-Memory MESI RFO Ping-Pong):**
  `MPI.Win.Allocate_shared` gave a window whose base is **128-B aligned** (`base % 64 = 0`, `base % 128 = 0`). The 4
  ranks run `host_shm_accumulate` (40,000,000 in-memory read-modify-writes each; LLVM loop `B34` has 5 `load i64` +
  5 `store i64`, no vectorisation, no calls). The trial time is that of the slowest rank, and every counter was
  bit-exact.

  | layout | lines touched | Colab T4 VM (2 vCPU) | ns/update | local Ryzen (6 C / 12 T) |
  |---|---|---:|---:|---:|
  | `stride = 1` (8 B) | `[0]` | **579,522,133 ns** | 14.488 | 247,898,400 ns |
  | `stride = 8` (64 B) | `[0,1,2,3]` | **192,220,911 ns** | 4.806 | 39,241,400 ns |
  | `stride = 16` (128 B) | `[0,2,4,6]` | **193,755,158 ns** | 4.844 | 31,647,300 ns |
  | **slowdown stride 1 / stride 8** | | **3.015×** | | **6.317×** |

  **MESI trace (Rank 0 → `shm_view[0]`, Rank 1 → `shm_view[1]`):** both ranks map the **same physical page**, so
  despite different virtual addresses both slots live in one physical 64-byte line L.
  (1) Both have L in **S** after earlier reads.
  (2) Rank 0's store retires into its store buffer and issues an upgrade (RFO/invalidate). Rank 1's copy goes
  **S → I**, and Rank 0's goes **S → E → M** when the store commits.
  (3) Rank 1's store finds L in **I** and issues a **Read-For-Ownership**. Rank 0 snoops it and supplies the dirty line
  (**M → I**), and Rank 1 goes **I → M**.
  (4) Rank 0's next iteration misses again (**I → RFO → M**, Rank 1 **M → I**), and so on: the line ping-pongs.
  Under x86-TSO stores drain in order, so a store waiting for ownership blocks every younger store behind it. The store
  buffer fills and the core stalls; only the loads are served by store-to-load forwarding.
  **What the two machines show:** on the Ryzen the 4 processes run on 4 real cores and the full cross-core RFO cost
  appears (**6.3×**, +5.2 ns per update). On Colab, `lscpu` shows a *single* physical core with 2 hyper-threads sharing
  one L1d, and 4 ranks time-slice on it. Two hyper-threads do not need MESI transfers to share an L1 line. The 3.0× still
  measured comes from the SMT equivalent: when one hyper-thread stores to a line the sibling has loads in flight from,
  the core performs memory-ordering machine clears (pipeline flushes). The guest's view of the topology may also
  differ from where the hypervisor actually places the vCPUs. Either way, a 64-byte collision costs 3–6× on both
  platforms.

- **Question 5 (128-Byte Spatial Prefetcher Pairing & Padding Trade-Offs):**
  Colab: `stride 8` **192,220,911 ns** vs `stride 16` **193,755,158 ns** → **1.008×** (best trials 191.39 vs 191.54 ms,
  1.0008×), so **no difference**. Ryzen: 39.24 vs 31.65 ms (0.806×), but its trials overlap (stride 8: 32.4–45.4 ms,
  stride 16: 27.8–34.9 ms) with unpinned processes, and the morning's pinned-thread lab found the opposite sign
  (1.105×, best-case 1.013×). So neither machine shows a reproducible 128-byte effect for this access pattern.
  **Mechanism:** the L2 adjacent-line (spatial) prefetcher treats memory as 128-byte-aligned pairs of 64-byte lines,
  even line *2k* with odd line *2k+1*. On a miss to one, it also requests the buddy. With 64-byte padding, ranks 0 and 1
  own lines 0 and 1 of the same pair (the window is 128-B aligned). A miss by rank 0 can prefetch line 1 out of rank 1's
  cache (M → S/I), and rank 1's next store needs a fresh RFO that invalidates the prefetched copy again: secondary false
  sharing at 128-byte granularity. 128-byte spacing gives each rank a whole pair. The prefetcher reacts to **misses**;
  our counters stay L1-resident in M state after the first touch, so it rarely fires, which is consistent with the null
  result. 64-byte padding starts to suffer when lines are evicted or when threads read around their slots.
  **Cost of excessive padding:** padding spends cache capacity on empty bytes. A 128-byte slot per 8-byte counter wastes
  94 % of each line pair. A 32 KiB L1d holds 512 lines, i.e. only **256** 128-byte slots. Padding every element of a
  per-thread array or struct this way multiplies its footprint by 16, overflowing L1 and then L2 (1 MiB here),
  causing **capacity misses**, more TLB pressure and wasted prefetch bandwidth. Pad only data that is *written* by
  different cores concurrently, and keep read-mostly data dense.

- **Question 6 (CFL Stability & Distributed Energy Dissipation):**
  Manifest: `kappa_max = 0.23999593165551458`, `Δx = 1/513 = 1.9493177388e-3`, `Δx² = 3.7998396e-6`.
  **`Δt_max = Δx² / (4·κ_max) = 3.7998396e-6 / 0.9599837 = 3.958233e-6`**, and the run used **`Δt = 0.9·Δt_max =
  3.562410e-6`**. The Courant/diffusion number is `κ_max·Δt/Δx² = 0.225 ≤ 0.25`, so the condition holds with a 10 %
  margin. The arithmetic-mean interface conductivities never exceed κ_max, and the cubic-damping stiffness adds only
  `3λu²_max·Δt = 3·0.15·2.039²·3.56e-6 = 6.7e-6`. 300 steps simulate `t = 1.068723e-3`.
  Global energy from `cart_comm.allreduce` (identical in serialized and overlapped modes):

  | step | E = Σ u Δx Δy | ΔE over the previous 100 steps | L2 step residual | max\|u\| |
  |---:|---:|---:|---:|---:|
  | 0 | 5.797908857210840e-02 | — | — | 2.0393171630 |
  | 100 | 5.797566933941159e-02 | −3.419233e-06 | 2.236201e-02 | 2.0001324015 |
  | 200 | 5.797241134241159e-02 | −3.257997e-06 | 2.133075e-02 | 1.9617821947 |
  | 300 | 5.796928947689404e-02 | −3.121866e-06 | 2.040906e-02 | 1.9242855861 |

  E falls strictly at every checkpoint (total −9.799e-6, −0.0169 %), and the step residual and max|u| fall monotonically,
  which indicates a stable, dissipative explicit scheme. **Budget check:** the flux-form diffusion terms cancel in pairs
  across every interior face, including the MPI halo faces, so `dE/dt = −λ∫u³ + ∫S − (wall flux)`. The seeded
  source/sink dipole integrates to `∫S dA = 2.8e-18` (zero), and the nearest pulse is ≈ 4.9σ from a wall, so
  `dE/dt ≈ −λ∫u³`. Two of this seed's pulses overlap (centres (0.609, 0.694) and (0.636, 0.718)), so ∫u³ must be
  evaluated on the summed field. That gives ∫u₀³ dA = 0.064687 and predicts **ΔE(0→100) ≈ −0.15 · 0.064687 · 100Δt =
  −3.457e-6**, against **−3.419e-6 measured (1.1 %)**. The shrinking decrements follow the decay of the peak (2.04 → 1.92).
