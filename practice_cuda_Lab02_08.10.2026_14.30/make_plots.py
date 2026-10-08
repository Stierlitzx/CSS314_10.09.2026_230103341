"""Render measured CUDA Lab 02 results into report images.

Reads nothing from disk: it imports the lab modules, runs the same
verifiable computations as verify_submission.py, times the CPU-fallback
kernels with time.perf_counter(), and exports four figures to plots/.

Run:  python make_plots.py   (from this folder)
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    plots = folder / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    import task1_divergence as t1
    import task2_stencil_1d as t2
    import task3_grid_stride as t3
    import task4_sobel_2d as t4

    # --- Task 1: real relative CPU-fallback timings (N=512, warm-up + 5 trials) ---
    rng = np.random.default_rng(230103341)
    host = rng.random(512).astype(np.float32)
    times = {}
    for name, kernel in (
        ("A_uniform", t1.kernel_a_uniform),
        ("B_divergence", t1.kernel_b_divergence),
        ("C_warp_aligned", t1.kernel_c_warp_aligned),
    ):
        buf = host.copy()
        kernel(buf, 512)  # warm-up (excludes import/JIT noise)
        trials = []
        for _ in range(5):
            buf = host.copy()
            t0 = time.perf_counter()
            kernel(buf, 512)
            trials.append(time.perf_counter() - t0)
        times[name] = float(np.median(trials))

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["A: Uniform", "B: Full divergence", "C: Warp-aligned"]
    values = [times["A_uniform"], times["B_divergence"], times["C_warp_aligned"]]
    bars = ax.bar(labels, values, color=["#2563eb", "#d97706", "#15803d"], width=0.55)
    ax.bar_label(bars, fmt="%.3f s", padding=5)
    ax.set(ylabel="Median wall time, s (N=512, 1000 it/elem, 5 trials)",
           title="Task 1: warp-divergence proxy timings (CPU fallback)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(plots / "task1_divergence.png", dpi=180)
    plt.close(fig)
    print("task1:", {k: round(v, 4) for k, v in times.items()})

    # --- Task 2: input sine vs smoothed output (N=10007, same as verifier) ---
    test_in = np.sin(np.linspace(0, 10, 10007)).astype(np.float32)
    h_gpu = t2.run_stencil(test_in)
    h_cpu = t2.cpu_stencil(test_in)
    delta = float(np.max(np.abs(h_gpu - h_cpu)))
    x = np.linspace(0, 10, 10007)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x[:600], test_in[:600], color="#94a3b8", linewidth=1.2, label="Input sin(x)")
    ax.plot(x[:600], h_gpu[:600], color="#2563eb", linewidth=2, label="Stencil output")
    ax.set(xlabel="x", ylabel="Value",
           title=f"Task 2: 3-point stencil smoothing (N=10007, max delta vs NumPy CPU = {delta:.2e})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "task2_stencil.png", dpi=180)
    plt.close(fig)
    print(f"task2: max delta = {delta:.3e}")

    # --- Task 3: grid-stride thread coverage over N=100,000 ---
    # 16,384 emulated threads sweep the array in passes; count elements/thread.
    N3 = 100_000
    n_threads = 256 * 64
    counts = np.zeros(n_threads, dtype=np.int64)
    for start in range(n_threads):
        c = 0
        i = start
        while i < N3:
            c += 1
            i += n_threads
        counts[start] = c
    uniq, freq = np.unique(counts, return_counts=True)
    fig, (bars_ax, cover_ax) = plt.subplots(1, 2, figsize=(11, 4.6))
    wedges = bars_ax.bar([f"{u} elems" for u in uniq], freq, color=["#d97706", "#15803d"], width=0.5)
    bars_ax.bar_label(wedges, padding=5)
    bars_ax.set(ylabel="Threads", title="Threads by element count")
    x = np.arange(n_threads)
    cover_ax.plot(x[:2000], counts[:2000], color="#2563eb", linewidth=0.8)
    cover_ax.set(xlabel="Emulated thread id (first 2000 of 16,384)",
                 ylabel="Elements handled", title="Per-thread coverage, N=100,000")
    cover_ax.grid(alpha=0.2)
    fig.suptitle("Task 3: 16,384-thread grid-stride covers 100,000 elements in 7 passes")
    fig.tight_layout()
    fig.savefig(plots / "task3_grid_stride.png", dpi=180)
    plt.close(fig)
    print("task3: coverage =", dict(zip(uniq.tolist(), freq.tolist())))

    # --- Task 4: ramp input + Sobel-X response side by side ---
    # Axis 0 varies (Sobel-X differentiates along that axis in this layout).
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1)).T
    sobel_res = t4.run_sobel(ramp)
    interior = float(np.max(np.abs(sobel_res[1:-1, 1:-1])))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6))
    im0 = left.imshow(ramp, cmap="gray", vmin=0, vmax=1)
    left.set_title("Task 4 input: horizontal ramp 64x64")
    fig.colorbar(im0, ax=left, fraction=0.046)
    vmax = max(float(np.max(np.abs(sobel_res))), 1e-9)
    # Zoom into the top-left 14x14 corner so the zeroed border row/col are visible.
    corner = sobel_res[:14, :14]
    im1 = right.imshow(corner, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for r in range(14):
        for c in range(14):
            right.text(c, r, f"{corner[r, c]:.2f}", ha="center", va="center",
                       fontsize=6, color="white" if abs(corner[r, c]) > vmax / 2 else "black")
    right.set_title(f"Sobel-X corner 14x14 (border zeros visible, interior {interior:.3f})")
    fig.colorbar(im1, ax=right, fraction=0.046)
    left.set_xticks([])
    left.set_yticks([])
    right.set_xticks(range(14))
    right.set_yticks(range(14))
    fig.tight_layout()
    fig.savefig(plots / "task4_sobel.png", dpi=180)
    plt.close(fig)
    print(f"task4: ramp interior max abs = {interior:.4f}")

    # ------------------------------------------------------------------
    # Result artifacts (actual kernel outputs, like result/mandelbrot_output.png)
    # ------------------------------------------------------------------
    results = folder / "result"
    results.mkdir(exist_ok=True)

    # Sobel-X on a synthetic scene: the displayed output IS run_sobel's return value.
    scene = np.full((128, 128), 0.1, dtype=np.float32)
    scene[20:60, 15:55] = 0.9                                  # bright square
    yy, xx = np.mgrid[0:128, 0:128]
    scene[(yy - 88) ** 2 + (xx - 88) ** 2 <= 24 ** 2] = 0.7    # disc
    scene[100:, :] = np.linspace(0.0, 1.0, 28, dtype=np.float32)[:, None]  # row gradient
    filtered = t4.run_sobel(scene)
    fmax = max(float(np.max(np.abs(filtered))), 1e-9)
    fig, (p_in, p_out, p_abs) = plt.subplots(1, 3, figsize=(13, 4.8))
    im_a = p_in.imshow(scene, cmap="gray", vmin=0, vmax=1)
    p_in.set_title("Synthetic input scene 128x128")
    fig.colorbar(im_a, ax=p_in, fraction=0.046)
    im_b = p_out.imshow(filtered, cmap="RdBu_r", vmin=-fmax, vmax=fmax)
    p_out.set_title("run_sobel(scene) output")
    fig.colorbar(im_b, ax=p_out, fraction=0.046)
    im_c = p_abs.imshow(np.abs(filtered), cmap="viridis", vmin=0, vmax=fmax)
    p_abs.set_title("|Sobel-X| edge strength")
    fig.colorbar(im_c, ax=p_abs, fraction=0.046)
    for p in (p_in, p_out, p_abs):
        p.set_xticks([])
        p.set_yticks([])
    fig.suptitle("Task 4 result: actual Sobel-X kernel output (borders zeroed)")
    fig.tight_layout()
    fig.savefig(results / "sobel_output.png", dpi=180)
    plt.close(fig)
    print(f"result: sobel_output.png, output range +/-{fmax:.3f}")

    # Stencil smoothing on a noisy signal: the blue curve IS run_stencil's return value.
    xs = np.linspace(0, 10, 2000).astype(np.float32)
    clean = (np.sin(xs) * 0.6 + 0.4 * np.sin(3 * xs)).astype(np.float32)
    noisy = clean + np.float32(0.15) * np.asarray(
        np.random.default_rng(230103341).standard_normal(2000), dtype=np.float32)
    smoothed = t2.run_stencil(noisy)
    rms_in = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rms_out = float(np.sqrt(np.mean((smoothed - clean) ** 2)))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(xs, noisy, color="#cbd5e1", linewidth=0.7, label="Noisy input")
    ax.plot(xs, smoothed, color="#2563eb", linewidth=1.8, label="run_stencil output")
    ax.plot(xs, clean, color="#334155", linewidth=1.0, linestyle="--", label="Clean signal")
    ax.set(xlabel="x", ylabel="Value",
           title=f"Task 2 result: stencil denoising, RMS error {rms_in:.3f} -> {rms_out:.3f}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results / "stencil_output.png", dpi=180)
    plt.close(fig)
    print(f"result: stencil_output.png, RMS {rms_in:.4f} -> {rms_out:.4f}")

    print(f"Generated four report images in {plots} and two result images in {results}")


if __name__ == "__main__":
    main()
