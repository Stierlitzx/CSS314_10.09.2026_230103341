"""Render README charts from the committed Colab T4 manifest. Never re-runs the solver."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
M = json.loads((ROOT / "submission_manifest.json").read_text(encoding="utf-8"))
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2563eb", "#ea580c", "#059669", "#7c3aed", "#0f172a", "#475569", "#cbd5e1"
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.edgecolor": "#94a3b8", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, alpha=0.6, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def frame(fig, title, subtitle, source):
    h = fig.get_figheight()
    fig.suptitle(title, x=0.06, y=1 - 0.1 / h, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 1 - 0.5 / h, subtitle, ha="left", va="top", fontsize=11, color=MUTED)
    fig.text(0.06, 0.1 / h, source, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def overlap_chart():
    steps = M["steps"]
    ser, ovl = M["serialized_ms"] / steps, M["overlapped_ms"] / steps
    ev = M["cuda_event_timings_ms"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.08, right=0.97, wspace=0.42)

    bars = a.barh(["Overlapped\n(2 streams)", "Serialized\n(1 stream)"], [ovl, ser], color=[BLUE, ORANGE],
                  height=0.55, edgecolor="white", linewidth=2)
    for rect, v in zip(bars, [ovl, ser]):
        a.text(v + 0.15, rect.get_y() + rect.get_height() / 2, f"{v:.2f} ms", va="center", color=INK, fontsize=11)
    a.set_xlim(0, ser * 1.25)
    a.set_xlabel("Wall time per step, slowest of 4 ranks (ms)")
    a.set_title(f"Overlap: {M['overlap_speedup']:.2f}x faster per step", loc="left")
    style(a, "x")

    names = ["Stage A: pack + D2H", "Interior kernel 254x254", "Stage C: Waitall + H2D\n+ unpack + ring",
             "Serialized step\n(exchange + full stencil)"]
    keys = ["overlapped_stageA_pack_d2h_ms", "overlapped_interior_kernel_ms", "overlapped_stageC_boundary_ms",
            "serialized_step_total_ms"]
    vals = [ev[k]["per_step_mean_ms"] * 1e3 for k in keys]
    colors = [BLUE, BLUE, BLUE, ORANGE]
    bars = b.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.55, edgecolor="white", linewidth=2)
    for rect, v in zip(bars, vals[::-1]):
        b.text(v + 120, rect.get_y() + rect.get_height() / 2, f"{v:,.0f} µs", va="center", color=INK, fontsize=10.5)
    b.set_xlim(0, max(vals) * 1.22)
    b.set_xlabel("CUDA-event time per step, mean of 4 ranks (µs)")
    b.set_title("Compute is tiny; waiting dominates", loc="left")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
             labels=["Overlapped pipeline stages", "Serialized step"], frameon=False, loc="upper right", fontsize=10)
    style(b, "x")
    frame(fig, "Dual-stream overlap on one shared Tesla T4",
          f"{steps} steps x 4 MPI ranks, 512x512 grid | results bit-identical in both modes (parity diff "
          f"{M['parity_max_diff']:.1f})", "Source: submission_manifest.json (Colab T4 run)")
    fig.savefig(PLOTS / "overlap_benchmark.png", dpi=160)
    plt.close(fig)


def energy_chart():
    tr = M["energy_trace_overlapped"]
    ser = M["energy_trace_serialized"]
    steps = [d["step"] for d in tr]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.09, right=0.98, wspace=0.42)
    panels = [("energy", "Global energy E = Σu·Δx·Δy", "E (allreduce)", 1.0),
              ("u_max", "Peak value max|u|", "max|u|", 1.0),
              ("l2_residual", "Step residual ‖uⁿ − uⁿ⁻¹‖₂", "L2 residual", 1.0)]
    for ax, (key, title, ylabel, _) in zip(axes, panels):
        pts = [(d["step"], d[key]) for d in tr if not (key == "l2_residual" and d["step"] == 0)]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=8, label="Overlapped")
        sp = [(d["step"], d[key]) for d in ser if not (key == "l2_residual" and d["step"] == 0)]
        ax.plot(*zip(*sp), linestyle="none", marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor=ORANGE, markeredgewidth=1.6, label="Serialized")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Time step")
        ax.set_ylabel(ylabel)
        ax.set_xticks(steps)
        ax.ticklabel_format(axis="y", useOffset=False)
        style(ax)
    axes[0].legend(frameon=False, loc="upper right", fontsize=10)
    e = [d["energy"] for d in tr]
    for i in range(1, len(e)):
        axes[0].annotate(f"ΔE −{(e[i - 1] - e[i]) * 1e6:.2f}e-6", xy=((steps[i - 1] + steps[i]) / 2, (e[i - 1] + e[i]) / 2),
                         xytext=(10, 4), textcoords="offset points", ha="left", fontsize=9, color=MUTED)
    frame(fig, "Stable, dissipative time stepping",
          f"Δt = {M['dt']:.4e} = 0.9·Δt_max (κ_max = {M['kappa_max']:.4f}) | every quantity falls monotonically; "
          "hollow rings = serialized run sitting exactly on the overlapped values",
          "Source: submission_manifest.json, energy_trace_* (Colab T4 run)")
    fig.savefig(PLOTS / "energy_stability.png", dpi=160)
    plt.close(fig)


def halo_traffic_chart():
    ranks, steps = M["mpi_ranks"], M["steps"]
    packed = 4 * 256 * 8 * ranks * steps
    span = (256 * 258 + 1) - (258 + 1) + 1
    extra = (span * 8 - 2048) * 2 * ranks * steps
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.27, right=0.95)
    labels = ["Packed strips\n(pack_halos_kernel)", "Unstrided bounding-box\ncopies of the 2 columns"]
    vals = [packed, packed + extra]
    bars = ax.barh(labels[::-1], vals[::-1], color=[ORANGE, BLUE], height=0.55, edgecolor="white", linewidth=2)
    ax.set_xscale("log")
    for rect, v in zip(bars, vals[::-1]):
        ax.text(v * 1.15, rect.get_y() + rect.get_height() / 2, f"{v / 1e6:,.1f} MB", va="center", color=INK)
    ax.set_xlim(1e6, 2e10)
    ax.set_xlabel("Device-to-host bytes over the whole run (log scale)")
    style(ax, "x")
    frame(fig, f"GPU packing moves {vals[1] / vals[0]:.0f}x less data over PCIe",
          f"{ranks} ranks x {steps} steps | a strided column is 256 doubles 2,064 B apart; a contiguous copy must span 526,328 B",
          "Computed from the array layout (258 x 258 float64, C order); see Q1")
    fig.savefig(PLOTS / "halo_traffic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    PLOTS.mkdir(exist_ok=True)
    overlap_chart()
    energy_chart()
    halo_traffic_chart()
    print("wrote", sorted(p.name for p in PLOTS.glob("*.png")))
