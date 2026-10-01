"""LAB PRACTICUM: OpenMP Multi-Core Scaling in Python (Numba)."""
import time
from pathlib import Path
import numpy as np
import numba
from numba import njit, prange

OUT_DIR = Path(__file__).resolve().parent.parent / "result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Hardware Threads Detected: {numba.config.NUMBA_NUM_THREADS}")

# ---------------- Challenge 1: Monte Carlo pi ----------------
@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / n_samples

_ = monte_carlo_pi(10_000)  # JIT warmup
SAMPLES = 120_000_000
thread_counts = sorted(set(t for t in [1, 2, 4, 8, numba.config.NUMBA_NUM_THREADS]
                           if t <= numba.config.NUMBA_NUM_THREADS))
print("\n--- Challenge 1: Monte Carlo Pi ---")
print(f"{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
print("-" * 55)
t1_baseline = None
mc_results = []
for t in thread_counts:
    numba.set_num_threads(t)
    start = time.perf_counter()
    pi_est = monte_carlo_pi(SAMPLES)
    elapsed = time.perf_counter() - start
    if t == 1:
        t1_baseline = elapsed
        speedup, efficiency = 1.0, 100.0
    else:
        speedup = t1_baseline / elapsed
        efficiency = (speedup / t) * 100.0
    mc_results.append((t, elapsed, speedup, efficiency))
    print(f"{t:<10} | {elapsed:<12.4f} | {speedup:<10.2f}x | {efficiency:<15.1f}")
print(f"pi estimate: {pi_est:.6f}")

# ---------------- Challenge 2: Mandelbrot ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

@njit(parallel=True)
def render_mandelbrot_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

_ = render_mandelbrot_rows(100, 100, 50)
_ = render_mandelbrot_cols(100, 100, 50)
H, W, MAX_IT = 2500, 2500, 1000
print("\n--- Challenge 2: Mandelbrot ---")
t0 = time.perf_counter()
grid_rows = render_mandelbrot_rows(H, W, MAX_IT)
t_rows = time.perf_counter() - t0
t1 = time.perf_counter()
grid_cols = render_mandelbrot_cols(H, W, MAX_IT)
t_cols = time.perf_counter() - t1
print(f"Row-Parallel Render Time:    {t_rows:.3f} s")
print(f"Column-Parallel Render Time: {t_cols:.3f} s")
plt.figure(figsize=(8, 8))
plt.imshow(grid_rows, cmap='magma', extent=[-2.0, 0.5, -1.2, 1.2])
plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
plt.axis('off')
plt.savefig(OUT_DIR / 'mandelbrot_output.png', dpi=300, bbox_inches='tight')
print(f"Saved image: {OUT_DIR / 'mandelbrot_output.png'}")

# ---------------- Challenge 3: Heat stencil ----------------
@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j]
            )

GRID_SIZE, STEPS = 1500, 300
print("\n--- Challenge 3: Heat Stencil (float64) ---")
def run_heat(dtype):
    u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=dtype)
    u_next = np.zeros_like(u)
    u[0, :] = 100.0
    u[:, 0] = 100.0
    u_next[0, :] = 100.0
    u_next[:, 0] = 100.0
    heat_step(u, u_next)
    start = time.perf_counter()
    for step in range(STEPS):
        heat_step(u, u_next)
        u, u_next = u_next, u
    return time.perf_counter() - start

t64 = run_heat(np.float64)
cells64 = (GRID_SIZE * GRID_SIZE * STEPS) / t64 / 1e6
print(f"Heat Diffusion Complete: {t64:.3f} s")
print(f"Throughput: {cells64:.2f} Megacells/sec")

print("--- Challenge 3: Heat Stencil (float32) ---")
t32 = run_heat(np.float32)
cells32 = (GRID_SIZE * GRID_SIZE * STEPS) / t32 / 1e6
print(f"Heat Diffusion Complete: {t32:.3f} s")
print(f"Throughput: {cells32:.2f} Megacells/sec")
print(f"Speedup (f64 -> f32): {t64 / t32:.2f}x")

with open(OUT_DIR / "benchmark_results.txt", "w") as f:
    f.write(f"Threads | Time(s) | Speedup | Efficiency(%)\n")
    for t, e, s, eff in mc_results:
        f.write(f"{t} | {e:.4f} | {s:.2f}x | {eff:.1f}\n")
    f.write(f"Mandelbrot rows: {t_rows:.3f} s\n")
    f.write(f"Mandelbrot cols: {t_cols:.3f} s\n")
    f.write(f"Heat float64: {t64:.3f} s, {cells64:.2f} Mcells/s\n")
    f.write(f"Heat float32: {t32:.3f} s, {cells32:.2f} Mcells/s\n")