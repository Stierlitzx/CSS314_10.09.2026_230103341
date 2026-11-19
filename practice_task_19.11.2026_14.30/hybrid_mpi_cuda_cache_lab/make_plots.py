"""Render README charts from the committed Colab manifest and the local Ryzen log. Never re-runs anything."""
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
M = json.loads((ROOT / "hybrid_lab_manifest.json").read_text(encoding="utf-8"))
RYZEN_LOG = (ROOT / "local_ryzen_shm_crosscheck.txt").read_text(encoding="utf-8")
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2563eb", "#ea580c", "#059669", "#7c3aed", "#0f172a", "#475569", "#cbd5e1"
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.edgecolor": "#94a3b8", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
STRIDES = (1, 8, 16)


def style(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, alpha=0.6, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def frame(fig, title, subtitle, source):
    h = fig.get_figheight()
    fig.suptitle(title, x=0.06, y=1 - 0.1 / h, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 1 - 0.5 / h, subtitle, ha="left", va="top", fontsize=11, color=MUTED)
    fig.text(0.06, 0.1 / h, source, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def ryzen_trials():
    out = {}
    for s in STRIDES:
        m = re.search(rf"stride_{s}\s.*?\n\s*slowest-rank trials \[([^\]]+)\]", RYZEN_LOG)
        out[s] = [int(v) for v in m.group(1).split(",")]
    return out


def shm_chart():
    iters = M["shm_iters"]
    colab = {s: M["shm_results"][f"stride_{s}"]["slowest_rank_ns_trials"] for s in STRIDES}
    machines = [("Colab VM: 1 core / 2 hyper-threads", colab), ("Local Ryzen 5 7640HS: 6 cores", ryzen_trials())]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    fig.subplots_adjust(top=0.77, bottom=0.17, left=0.07, right=0.98, wspace=0.28)
    colors = {1: ORANGE, 8: BLUE, 16: BLUE}
    for ax, (title, data) in zip(axes, machines):
        med8 = float(np.median(data[8])) / iters
        top = 0
        for i, s in enumerate(STRIDES):
            per = [t / iters for t in data[s]]
            med = float(np.median(per))
            ax.bar(i, med, width=0.6, color=colors[s], edgecolor="white", linewidth=2, zorder=2,
                   hatch="//" if s == 16 else None)
            ax.scatter([i] * len(per), per, s=28, color="white", edgecolor=INK, linewidth=0.8, zorder=3)
            ax.text(i, max(per) * 1.03 + 0.1, f"{med:.2f} ns\n{med / med8:.2f}x", ha="center", va="bottom",
                    fontsize=10.5, color=INK)
            top = max(top, max(per))
        ax.set_xticks(range(3), [f"stride {s}\n{s * 8} B apart" for s in STRIDES])
        ax.set_ylim(0, top * 1.3)
        ax.set_ylabel("ns per update, slowest rank (median)")
        ax.set_title(title, loc="left")
        style(ax)
    axes[1].legend([Rectangle((0, 0), 1, 1, color=ORANGE), Rectangle((0, 0), 1, 1, color=BLUE),
                    Rectangle((0, 0), 1, 1, facecolor=BLUE, hatch="//", edgecolor="white")],
                   ["4 ranks share 1 line", "64-B padding", "128-B padding"], frameon=False, loc="upper right",
                   fontsize=10)
    frame(fig, "False sharing through an MPI-3 shared window",
          f"4 MPI ranks x {iters:,} updates on Win.Allocate_shared counters, 3 trials (bar = median, dots = trials) | "
          f"Colab {M['shm_false_sharing_slowdown_stride1_vs_stride8']:.2f}x vs Ryzen "
          f"{float(np.median(machines[1][1][1])) / float(np.median(machines[1][1][8])):.2f}x",
          "Sources: hybrid_lab_manifest.json (Colab T4 VM) · local_ryzen_shm_crosscheck.txt (Tier-4 numbers only)")
    fig.savefig(PLOTS / "shm_false_sharing.png", dpi=160)
    plt.close(fig)


def gpu_chart():
    steps = M["stencil_steps"]
    ser, ovl = M["gpu_serialized_ms"] / steps, M["gpu_overlapped_ms"] / steps
    ev = M["cuda_event_timings_ms"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.08, right=0.97, wspace=0.42)
    bars = a.barh(["Overlapped\n(2 streams)", "Serialized\n(1 stream)"], [ovl, ser], color=[BLUE, ORANGE],
                  height=0.55, edgecolor="white", linewidth=2)
    for rect, v in zip(bars, [ovl, ser]):
        a.text(v + 0.08, rect.get_y() + rect.get_height() / 2, f"{v:.2f} ms", va="center", color=INK, fontsize=11)
    a.set_xlim(0, max(ser, ovl) * 1.3)
    a.set_xlabel("Wall time per step, slowest of 4 ranks (ms)")
    a.set_title(f"Overlap: {M['gpu_overlap_speedup']:.2f}x (slower here)", loc="left")
    style(a, "x")
    names = ["Pack + D2H", "Interior kernel 254x254", "Waitall + H2D\n+ unpack + ring", "Serialized step"]
    keys = ["overlapped_pack_d2h_ms", "overlapped_interior_kernel_ms", "overlapped_boundary_stageC_ms",
            "serialized_step_total_ms"]
    vals = [ev[k]["mean_over_ranks"] / steps * 1e3 for k in keys]
    bars = b.barh(names[::-1], vals[::-1], color=[ORANGE, BLUE, BLUE, BLUE], height=0.55, edgecolor="white",
                  linewidth=2)
    for rect, v in zip(bars, vals[::-1]):
        b.text(v + 60, rect.get_y() + rect.get_height() / 2, f"{v:,.0f} µs", va="center", color=INK, fontsize=10.5)
    b.set_xlim(0, max(vals) * 1.25)
    b.set_xlabel("CUDA-event time per step, mean of 4 ranks (µs)")
    b.set_title("The GPU kernel is 0.3 ms; the rest is host waiting", loc="left")
    b.legend([Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=ORANGE)],
             ["Overlapped pipeline stages", "Serialized step"], frameon=False, loc="upper right", fontsize=10)
    style(b, "x")
    frame(fig, "Dual-stream overlap on a 2-vCPU host",
          f"{steps} steps x 4 ranks sharing one Tesla T4 and 2 vCPUs | parity diff {M['parity_max_diff']:.1f} "
          "(bit-identical) | extra launches and busy-polling Waitall cost more than the hidden 0.3 ms",
          "Source: hybrid_lab_manifest.json (Colab T4 run)")
    fig.savefig(PLOTS / "gpu_overlap.png", dpi=160)
    plt.close(fig)


def energy_chart():
    tr = M["energy_trace_overlapped"]
    steps = [d["step"] for d in tr]
    e = [d["energy"] for d in tr]
    p = M["seeded_parameters"]
    h = M["dx"]
    g = np.arange(1, M["global_grid"][0] + 1) * h
    Y, X = np.meshgrid(g, g, indexing="ij")
    u0 = sum(a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
             for cx, cy, a, s in zip(p["pulse_cx"], p["pulse_cy"], p["pulse_amp"], p["pulse_sigma"]))
    predicted = M["lambda_damp"] * float((u0 ** 3).sum() * h * h) * 100 * M["dt"]
    measured = [e[i - 1] - e[i] for i in range(1, len(e))]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.0))
    fig.subplots_adjust(top=0.77, bottom=0.17, left=0.09, right=0.98, wspace=0.33)
    a.plot(steps, e, color=BLUE, linewidth=2, marker="o", markersize=8, label="Overlapped")
    a.plot(steps, [d["energy"] for d in M["energy_trace_serialized"]], linestyle="none", marker="o", markersize=13,
           markerfacecolor="none", markeredgecolor=ORANGE, markeredgewidth=1.6, label="Serialized")
    a.set_xticks(steps)
    a.ticklabel_format(axis="y", useOffset=False)
    a.set_xlabel("Time step")
    a.set_ylabel("E = Σu·Δx·Δy (cart_comm.allreduce)")
    a.set_title("Global energy falls every 100 steps", loc="left")
    a.legend(frameon=False, loc="upper right", fontsize=10)
    style(a)

    labels = [f"{steps[i - 1]}→{steps[i]}" for i in range(1, len(steps))]
    x = np.arange(len(labels))
    b.bar(x, [m * 1e6 for m in measured], width=0.55, color=BLUE, edgecolor="white", linewidth=2, label="Measured −ΔE")
    b.scatter([0], [predicted * 1e6], marker="D", s=70, color=ORANGE, zorder=3,
              label="Predicted λ∫u₀³·100Δt")
    for xi, m in zip(x, measured):
        b.text(xi, m * 1e6 / 2, f"{m * 1e6:.3f}", ha="center", va="center", fontsize=11, color="white",
               fontweight="bold")
    b.annotate(f"{predicted * 1e6:.3f} predicted\n({abs(predicted - measured[0]) / measured[0] * 100:.1f}% off)",
               xy=(0, predicted * 1e6), xytext=(0.32, predicted * 1e6 + 0.35), fontsize=9.5, color=MUTED,
               arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    b.set_xticks(x, labels)
    b.set_ylim(0, max(measured) * 1e6 * 1.35)
    b.set_xlabel("Step interval")
    b.set_ylabel("Energy lost per 100 steps (×1e-6)")
    b.set_title("The loss matches the cubic damping term", loc="left")
    b.legend(frameon=False, loc="upper right", fontsize=10)
    style(b)
    frame(fig, "Stable, dissipative and physically consistent",
          f"Δt = {M['dt']:.4e} = 0.9·Δt_max (κ_max = {M['kappa_max']:.4f}, κΔt/Δx² = 0.225) | "
          f"∫S dA ≈ 0, so dE/dt ≈ −λ∫u³",
          "Source: hybrid_lab_manifest.json (energy traces, seeded pulse parameters)")
    fig.savefig(PLOTS / "energy_budget.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    PLOTS.mkdir(exist_ok=True)
    shm_chart()
    gpu_chart()
    energy_chart()
    print("wrote", sorted(p.name for p in PLOTS.glob("*.png")))
