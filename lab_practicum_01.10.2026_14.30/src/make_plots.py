"""Generate plots for the OpenMP/Numba lab practicum from the recorded results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# ---------------- Challenge 1: scaling ----------------
threads = np.array([1, 2, 4, 8, 12])
times = np.array([2.2401, 1.3111, 0.9896, 0.5781, 0.5822])
speedup = times[0] / times
efficiency = speedup / threads * 100.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(threads, times, "o-", label="measured")
ax1.set_xlabel("Threads"); ax1.set_ylabel("Time (s)")
ax1.set_title("Monte Carlo pi: latency vs threads")
ax1.grid(alpha=0.3); ax1.legend()
ax2.plot(threads, speedup, "o-", label="speedup")
ax2.plot(threads, threads, "--", label="ideal linear", color="gray")
ax2.set_xlabel("Threads"); ax2.set_ylabel("Speedup (x)")
ax2b = ax2.twinx()
ax2b.plot(threads, efficiency, "s--", color="tab:red", label="efficiency")
ax2b.set_ylabel("Efficiency (%)", color="tab:red")
ax2b.tick_params(axis="y", colors="tab:red")
ax2.set_title("Speedup vs ideal (Amdahl)"); ax2.grid(alpha=0.3)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS / "challenge1_scaling.png", dpi=150)
plt.close(fig)

# ---------------- Challenge 2: axis decomposition ----------------
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["rows (prange over r)", "cols (prange over c)"],
              [1.441, 1.397], color=["tab:blue", "tab:orange"])
ax.bar_label(bars, fmt="%.3f s")
ax.set_ylabel("Render time (s)")
ax.set_title(f"Mandelbrot 2500x2500, max_iter=1000 (12 threads)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(PLOTS / "challenge2_axis.png", dpi=150)
plt.close(fig)

# ---------------- Challenge 3: bandwidth ----------------
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["float64", "float32"], [1540.59, 2026.62],
              color=["tab:blue", "tab:green"])
ax.bar_label(bars, fmt="%.0f")
ax.set_ylabel("Throughput (Megacells/s)")
ax.set_title("Heat stencil 1500x1500 x 300 steps (12 threads)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(PLOTS / "challenge3_bandwidth.png", dpi=150)
plt.close(fig)

# ---------------- Mandelbrot render ----------------
@njit(parallel=True)
def render_mandelbrot_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

_ = render_mandelbrot_rows(100, 100, 50)  # JIT warmup
grid = render_mandelbrot_rows(2500, 2500, 1000)
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="magma", extent=[-2.0, 0.5, -1.2, 1.2])
plt.title("Mandelbrot 2500x2500 (Numba prange, 12 threads)")
plt.axis("off")
fig.savefig(PLOTS / "mandelbrot.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Plots written to {PLOTS}")
