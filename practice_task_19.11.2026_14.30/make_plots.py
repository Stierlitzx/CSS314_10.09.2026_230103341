"""Render README charts from the committed cache_lab_manifest.json. Never re-runs the benchmark."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
M = json.loads((ROOT / "cache_lab_manifest.json").read_text(encoding="utf-8"))
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2563eb", "#ea580c", "#059669", "#7c3aed", "#0f172a", "#475569", "#cbd5e1"
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.edgecolor": "#94a3b8", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
ITERS = M["iterations_per_thread"]


def style(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, alpha=0.6, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def frame(fig, title, subtitle, source):
    h = fig.get_figheight()
    fig.suptitle(title, x=0.06, y=1 - 0.1 / h, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 1 - 0.5 / h, subtitle, ha="left", va="top", fontsize=11, color=MUTED)
    fig.text(0.06, 0.1 / h, source, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def stride_sweep_chart():
    sr = M["stride_results"]
    cases = [("1 thread\nalone", M["single_thread_reference"]["wall_ns_trials"], PURPLE),
             ("stride 1\n8 B apart", sr["stride_1"]["wall_ns_trials"], ORANGE),
             ("stride 4\n32 B apart", sr["stride_4"]["wall_ns_trials"], ORANGE),
             ("stride 8\n64 B apart", sr["stride_8"]["wall_ns_trials"], BLUE),
             ("stride 16\n128 B apart", sr["stride_16"]["wall_ns_trials"], BLUE),
             ("register\nprivate", M["private_baseline"]["wall_ns_trials"], GREEN)]
    base = sorted(sr["stride_8"]["wall_ns_trials"])[len(sr["stride_8"]["wall_ns_trials"]) // 2] / ITERS
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.08, right=0.98)
    for i, (label, trials, color) in enumerate(cases):
        per = [t / ITERS for t in trials]
        med = sorted(per)[len(per) // 2]
        ax.bar(i, med, width=0.62, color=color, edgecolor="white", linewidth=2, zorder=2)
        ax.scatter([i] * len(per), per, s=26, color="white", edgecolor=INK, linewidth=0.8, zorder=3)
        ax.text(i, max(per) + 0.12, f"{med:.3f} ns\n{med / base:.2f}x stride 8", ha="center", va="bottom",
                fontsize=10, color=INK)
    ax.set_xticks(range(len(cases)), [c[0] for c in cases])
    ax.set_ylabel("ns per update per thread (median)")
    ax.set_ylim(0, 7.3)
    style(ax)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE, GREEN, PURPLE)]
    ax.legend(handles, ["False sharing (counters share a line)", "Padded (one line per thread)",
                        "Register-private control", "Single-thread reference"],
              frameon=False, loc="upper right", fontsize=10)
    frame(fig, f"Sharing one cache line makes the same work {M['slowdown_stride1_vs_stride8']:.1f}x slower",
          f"4 pinned threads x {ITERS:,} read-modify-write updates, 5 trials each (bar = median, dots = trials) | "
          "Ryzen 5 7640HS, on battery",
          "Source: cache_lab_manifest.json")
    fig.savefig(PLOTS / "stride_sweep.png", dpi=160)
    plt.close(fig)


def cache_line_map():
    line = M["cache_line_size"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    fig.subplots_adjust(top=0.74, bottom=0.17, left=0.13, right=0.98)
    strides = [1, 4, 8, 16]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    max_b = 8 * 16 * 3 + 64
    for x0 in range(0, max_b, 4 * line):            # shade every other 128-B prefetch pair
        ax.add_patch(Rectangle((x0, -0.6), line * 2, len(strides) + 0.2, color="#f1f5f9", zorder=0))
    for x in range(0, max_b + 1, line):
        ax.axvline(x, color="#94a3b8", linewidth=0.8, zorder=1)
        if x < max_b:
            ax.text(x + line / 2, len(strides) - 0.35, f"line {x // line}", ha="center", fontsize=9, color=MUTED)
    for row, s in enumerate(strides):
        y = len(strides) - 1 - row
        lines_used = sorted({t * s * 8 // line for t in range(4)})
        for t in range(4):
            off = t * s * 8
            ax.add_patch(Rectangle((off, y - 0.28), 8, 0.56, color=colors[t], zorder=3))
            if s > 1:
                ax.text(off + 4, y + 0.34, f"T{t}", ha="center", fontsize=9, color=INK, zorder=4)
        if s == 1:
            ax.text(16, y + 0.34, "T0–T3", ha="center", fontsize=9, color=INK, zorder=4)
        n = len(lines_used)
        note = "1 shared line" if n == 1 else (f"{n} lines, 2 threads each" if n < 4 else f"{n} lines, 1 per thread")
        ax.text(max_b + 6, y, note, va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(strides)), [f"stride {s}\n({s * 8} B)" for s in strides][::-1])
    ax.set_xlim(-4, max_b + 150)
    ax.set_ylim(-0.6, len(strides) - 0.1)
    ax.set_xticks(range(0, max_b + 1, line))
    ax.set_xlabel("Byte offset from the 128-byte-aligned buffer base")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    frame(fig, "Where each thread's 8-byte counter lands",
          f"{line}-byte cache lines; shaded bands = 128-byte adjacent-line prefetch pairs | T0..T3 = threads 0..3",
          "Layouts from the lab spec; byte offsets = thread x stride x 8")
    fig.savefig(PLOTS / "cache_line_map.png", dpi=160)
    plt.close(fig)


def gil_chart():
    g = M["gil_bytecode_baseline"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(top=0.77, bottom=0.17, left=0.07, right=0.97, wspace=0.42)
    ts = [1, 2, 4]
    for stride, color in ((1, ORANGE), (8, BLUE)):
        ys = [g[f"stride_{stride}_T{t}"]["wall_ns"] / 1e6 for t in ts]
        a.plot(ts, ys, color=color, linewidth=2, marker="o", markersize=8, label=f"stride {stride}")
    a.text(4.1, g["stride_1_T4"]["wall_ns"] / 1e6, "both\n≈360 ms", va="center", fontsize=10, color=INK)
    t1 = g["stride_1_T1"]["wall_ns"] / 1e6
    a.plot(ts, [t1] * 3, color=MUTED, linewidth=1.5, linestyle="--", label="ideal if threads ran in parallel")
    a.set_xticks(ts)
    a.set_xlim(0.8, 4.8)
    a.set_xlabel("Python threads (each does 300,000 updates)")
    a.set_ylabel("Wall time (ms)")
    a.set_ylim(0, 420)
    a.set_title("With the GIL, more threads = more time", loc="left")
    a.legend(frameon=False, loc="upper left", fontsize=10)
    style(a)

    sr = M["stride_results"]
    labels = ["Python bytecode\n4 threads (GIL)", "Numba nogil\nstride 1 (false sharing)", "Numba nogil\nstride 8 (padded)"]
    vals = [g["stride_1_T4"]["ns_per_update_aggregate"],
            sr["stride_1"]["wall_ns_median"] / (4 * ITERS), sr["stride_8"]["wall_ns_median"] / (4 * ITERS)]
    bars = b.barh(labels[::-1], vals[::-1], color=[BLUE, ORANGE, PURPLE], height=0.55, edgecolor="white", linewidth=2)
    b.set_xscale("log")
    for rect, v in zip(bars, vals[::-1]):
        b.text(v * 1.25, rect.get_y() + rect.get_height() / 2, f"{v:,.3g} ns", va="center", color=INK, fontsize=10.5)
    b.set_xlim(0.05, 3000)
    b.set_xlabel("Aggregate ns per update, 4 threads (log scale)")
    b.set_title("Interpreter cost hides the cache effect", loc="left")
    style(b, "x")
    frame(fig, "The GIL masks false sharing; nogil=True exposes it",
          "Left: pure-Python threads never run concurrently, so layout barely matters (stride 1 vs 8 at T=4: 361 vs 360 ms)",
          "Source: cache_lab_manifest.json (gil_bytecode_baseline, stride_results)")
    fig.savefig(PLOTS / "gil_vs_nogil.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    PLOTS.mkdir(exist_ok=True)
    stride_sweep_chart()
    cache_line_map()
    gil_chart()
    print("wrote", sorted(p.name for p in PLOTS.glob("*.png")))
