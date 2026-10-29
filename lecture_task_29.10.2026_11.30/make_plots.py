"""Render retained empirical results; never run benchmarks or rewrite their data."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "lab-submission" / "results"
EVIDENCE = ROOT / "evidence"
PLOTS = ROOT / "plots"
BLUE, ORANGE, GREEN, PURPLE, INK = "#2563eb", "#ea580c", "#059669", "#7c3aed", "#0f172a"
INPUTS = [RESULTS / name for name in ("host_profile.json", "salted_params.json", "stride_results.tsv", "loop_interchange.json",
          "false_sharing_karp_flatt.tsv", "scaling_metrics.tsv", "roofline_summary.json")]
INPUTS += [EVIDENCE / "dense_outputs.npz", EVIDENCE / "stencil_output.npy"]


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def read_table(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def hashes():
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUTS}


def median_cases(rows, mode):
    cases = {}
    for p in sorted({int(row["P"]) for row in rows if row["mode"] == mode}):
        group = [row for row in rows if row["mode"] == mode and int(row["P"]) == p]
        median = statistics.median(int(row["time_ns"]) for row in group)
        if not all(float(row["median_time_ns"]) == median for row in group):
            raise ValueError(f"Stored median disagrees with raw {mode}, P={p} samples")
        cases[p] = {"median_ns": median, "runs_ns": [int(row["time_ns"]) for row in group], "row": group[0]}
    return cases


def axis_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def worker_axis(ax, counts):
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Worker processes P")


def finish(fig, filename, title, subtitle, footer):
    fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=18, weight="bold", color=INK)
    fig.text(0.07, 0.90, subtitle, fontsize=10, color="#475569")
    fig.text(0.07, 0.025, footer, fontsize=9, color="#475569")
    fig.subplots_adjust(top=0.79, bottom=0.18, left=0.08, right=0.97, wspace=0.30)
    fig.savefig(PLOTS / filename, dpi=170, facecolor="white")
    plt.close(fig)


def locality(strides, dense, settings):
    fig, (latency, timing) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = [int(row["stride"]) for row in strides]
    y = [float(row["ns_per_element"]) for row in strides]
    for row in strides:
        touched = int(row["elements_touched"])
        raw = [int(row[f"run_{i}_ns"])/touched for i in range(1, 6)]
        if not np.isclose(statistics.median(raw), float(row["ns_per_element"]), rtol=1e-12):
            raise ValueError("Stride chart median does not match raw samples")
        latency.scatter([int(row["stride"])]*5, raw, color=BLUE, alpha=0.20, s=17)
    latency.plot(x, y, "o-", color=BLUE, linewidth=2.5, label="Median of 5 samples")
    latency.axvline(8, linestyle="--", color=ORANGE, alpha=0.8, label="64-byte step (stride 8)")
    latency.annotate(f"{y[x.index(16)]:.2f} ns at stride 16", (16, y[x.index(16)]),
                     xytext=(5, 6.5), arrowprops={"arrowstyle": "->", "color": INK}, fontsize=10)
    worker_axis(latency, x)
    latency.set(xlabel="Stride (float64 elements)", ylabel="Nanoseconds per element read",
                title="Read locality | lower is better", ylim=(0, max(y)*1.30))
    latency.legend(loc="upper left", fontsize=8.5)
    values = [dense["median_ns"][name]/1e6 for name in ("ijk", "ikj")]
    bars = timing.bar([0, 1], values, color=[ORANGE, GREEN], width=0.55)
    timing.bar_label(bars, labels=[f"{v:.2f} ms" for v in values], padding=7, weight="bold")
    for index, name in enumerate(("ijk", "ikj")):
        if statistics.median(dense["runs_ns"][name]) != dense["median_ns"][name]:
            raise ValueError("Dense timing median mismatch")
        samples = np.array(dense["runs_ns"][name])/1e6
        timing.scatter(index+np.linspace(-0.08, 0.08, len(samples)), samples, color=INK, s=19, zorder=4)
    timing.set(xticks=[0, 1], xticklabels=["Naive i-j-k", "Interchanged i-k-j"],
               ylabel="Elapsed time (ms)", ylim=(0, max(values)*1.28),
               title=f"Loop interchange | {dense['speedup']:.2f}x speedup")
    for ax in (latency, timing):
        axis_style(ax)
    finish(fig, "memory_locality.png", "M1 | Memory layout changes execution time",
           f"Student {settings['student_id']} | 128 MiB stride buffer | M={dense['M']} sub-block, row pitch N={dense['N']}",
           "Sources: stride_results.tsv + loop_interchange.json | Dots retain individual timing samples; bars/lines show medians.")


def sharing(rows):
    fig, (timing, metric) = plt.subplots(1, 2, figsize=(12.5, 5.3))
    counts = [1, 2, 4, 8]
    groups = {mode: median_cases(rows, mode) for mode in ("unpadded", "padded", "privatized")}
    for mode, color, label in (("unpadded", ORANGE, "Unpadded"), ("padded", BLUE, "64-byte padded"),
                               ("privatized", GREEN, "Privatized")):
        values = [groups[mode][p]["median_ns"]/1e6 for p in counts]
        timing.plot(counts, values, "o-", color=color, label=label, linewidth=2.3)
        for p in counts:
            raw = np.array(groups[mode][p]["runs_ns"])/1e6
            timing.scatter([p]*len(raw), raw, color=color, alpha=0.24, s=20)
        if mode != "privatized":
            e = [float(groups[mode][p]["row"]["karp_flatt_handout"]) for p in counts[1:]]
            metric.plot(counts[1:], e, "o-", color=color, label=label, linewidth=2.3)
    ratio = groups["unpadded"][8]["median_ns"]/groups["padded"][8]["median_ns"]
    timing.set(title=f"Padding at P=8: {ratio:.2f}x faster", ylabel="Elapsed time (ms)", ylim=(0, None))
    metric.axhline(1, color="#64748b", linestyle="--", label="Ideal fixed-K-per-worker baseline: 1")
    metric.set(title="Requested Karp-Flatt metric", ylabel="e(P) = (P TP / T1 - 1) / (P - 1)", yscale="log")
    worker_axis(timing, counts)
    worker_axis(metric, counts[1:])
    for ax in (timing, metric):
        axis_style(ax)
        ax.axvspan(6, 8.6, color=PURPLE, alpha=0.07)
        ax.set_xlim(0.85 if ax is timing else 1.7, 8.6)
        ax.legend(fontsize=8, loc="upper left")
    metric.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.52))
    finish(fig, "false_sharing.png", "M2 | Cache-line contention and recovery",
           "5,000,000 increments per worker | Three samples per case | Shaded region exceeds six physical cores",
           "Source: false_sharing_karp_flatt.tsv | Total work grows with P; e(P) is not a fixed-work serial fraction.")


def scaling(rows, roofline):
    fig, (speedup, efficiency) = plt.subplots(1, 2, figsize=(12.5, 5.3))
    strong, weak = median_cases(rows, "strong"), median_cases(rows, "weak")
    counts = list(strong)
    measured = [strong[1]["median_ns"]/strong[p]["median_ns"] for p in counts]
    s = roofline["amdahl_fit"]["fitted_serial_fraction"]
    domain = np.linspace(1, max(counts), 150)
    speedup.plot(domain, domain, "--", color="#94a3b8", label="Ideal linear speedup")
    speedup.plot(domain, 1/(s+(1-s)/domain), color=ORANGE, linewidth=2, label=f"Amdahl fit: s={100*s:.2f}%")
    speedup.plot(counts, measured, "o-", color=BLUE, linewidth=2.5, label="Measured strong scaling")
    speedup.set(title=f"Fixed 1088 x 1088 | {measured[-1]:.2f}x at {counts[-1]} cores",
                ylabel="Speedup T1 / TP", ylim=(0, max(counts)*1.15))
    for p, value in zip(counts, measured):
        speedup.annotate(f"{value:.2f}x", (p, value), xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9)
    strong_e = [100*value/p for p, value in zip(counts, measured)]
    weak_e = [100*weak[1]["median_ns"]/weak[p]["median_ns"] for p in counts]
    efficiency.axhline(100, color="#94a3b8", linestyle="--", label="Ideal efficiency")
    efficiency.plot(counts, strong_e, "o-", color=BLUE, linewidth=2.5, label="Strong: T1 / (P TP)")
    efficiency.plot(counts, weak_e, "s-", color=PURPLE, linewidth=2.5, label="Weak: T1 / TP")
    efficiency.annotate(f"{weak_e[-1]:.2f}%", (counts[-1], weak_e[-1]), xytext=(-12, 13),
                        textcoords="offset points", ha="right", color=PURPLE, weight="bold")
    efficiency.set(title="Growing the working set reduces weak efficiency", ylabel="Efficiency (%)", ylim=(0, 115))
    for ax in (speedup, efficiency):
        axis_style(ax)
        ax.set(xticks=counts, xlabel="Worker processes P", xlim=(0.7, max(counts)+0.4))
        ax.legend(fontsize=8.5, loc="lower right")
    finish(fig, "scaling.png", "M3 | Strong and weak scaling diverge",
           "Median of three samples | 100 independent stencil updates per sample | P=1, 2, 4, 6 physical cores",
           "Source: scaling_metrics.tsv + roofline_summary.json | Amdahl is a fitted model; weak dimensions use ceil(N sqrt(P)).")


def roofline_chart(model):
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ai = np.geomspace(0.015, 30, 500)
    compute = model["theoretical_peak_compute_gflops"]
    memory = model["theoretical_peak_memory_bandwidth_gb_s"]
    copy = model["bandwidth_calibration"]["median_gb_s"]
    intensity = model["ideal_streaming_arithmetic_intensity_flop_per_byte"]
    performance = model["best_measured_stencil_gflops"]
    ax.loglog(ai, np.minimum(compute, memory*ai), color=BLUE, linewidth=2.8,
              label=f"Theoretical Roofline: {memory:.1f} GB/s, {compute:.1f} GFLOP/s compute model")
    ax.loglog(ai, np.minimum(compute, copy*ai), color=GREEN, linestyle="--", linewidth=2,
              label=f"Single-process copy reference: {copy:.2f} GB/s (not a universal bound)")
    ax.axvline(model["ridge_point_flop_per_byte"], color=ORANGE, linestyle=":", linewidth=1.7,
               label=f"Theoretical ridge: {model['ridge_point_flop_per_byte']:.2f} FLOP/B")
    ax.scatter([intensity], [performance], color=PURPLE, marker="D", s=90, zorder=5,
               label="Measured stencil throughput at assumed streaming AI")
    ax.annotate(f"Stencil: {performance:.2f} GFLOP/s\nAssumed AI = {intensity:.4f} FLOP/B",
                (intensity, performance), xytext=(0.065, 100), color=PURPLE, weight="bold",
                arrowprops={"arrowstyle": "->", "color": PURPLE}, fontsize=10)
    ax.set(xlim=(0.015, 30), ylim=(1, compute*1.8),
           xlabel="Arithmetic intensity (FLOP / modeled DRAM byte, log scale)",
           ylabel="Performance (GFLOP/s, log scale)")
    axis_style(ax)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    finish(fig, "roofline.png", "M3 | Positioning the stencil on the Roofline",
           "Five-point stencil: 5 FLOPs/cell | 0.104 FLOP/B without reuse; 0.3125 FLOP/B with ideal streaming reuse",
           "Source: roofline_summary.json | The point's AI is modeled, not measured by counters; cache reuse can exceed the copy reference.")


def numerical_outputs():
    with np.load(EVIDENCE / "dense_outputs.npz", allow_pickle=False) as saved:
        matrix = saved["ikj"]
        difference = saved["ikj"]-saved["reference"]
        np.testing.assert_allclose(saved["ijk"], matrix, rtol=0, atol=1e-12)
    stencil = np.load(EVIDENCE / "stencil_output.npy", allow_pickle=False)
    max_error = float(np.max(np.abs(difference)))
    if max_error > 1e-12:
        raise ValueError("Saved matrix exceeds the required numerical tolerance")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.1))
    datasets = [(matrix, "Dense C: interchanged loop", "viridis", None),
                (difference, "C - NumPy reference", "coolwarm", max_error),
                (stencil, "Stencil output: six physical cores", "viridis", None)]
    for ax, (data, title, cmap, limit) in zip(axes, datasets):
        image = ax.imshow(data, cmap=cmap, interpolation="nearest",
                          **({"vmin": -limit, "vmax": limit} if limit else {}))
        ax.set(title=title, xlabel="Column", ylabel="Row")
        ax.set_title(title, fontsize=10)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.03)
        colorbar.ax.tick_params(labelsize=8)
    finish(fig, "numerical_outputs.png", "M4 | Numerical verification",
           f"Saved matrix and stencil arrays | Maximum dense reference error: {max_error:.2e} | Absolute tolerance: 1e-12",
           "Sources: evidence/dense_outputs.npz + stencil_output.npy | Colormaps show values, not timing or cache-hit measurements.")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": "#475569", "ytick.color": "#475569",
                         "axes.edgecolor": "#cbd5e1", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    original = hashes()
    settings, dense, model = read_json("salted_params.json"), read_json("loop_interchange.json"), read_json("roofline_summary.json")
    PLOTS.mkdir(exist_ok=True)
    locality(read_table("stride_results.tsv"), dense, settings)
    sharing(read_table("false_sharing_karp_flatt.tsv"))
    scaling(read_table("scaling_metrics.tsv"), model)
    roofline_chart(model)
    numerical_outputs()
    if hashes() != original:
        raise RuntimeError("A source result changed while plotting")
    references = re.findall(r"!\[[^\]]*\]\((plots/[^)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8"))
    if len(references) != 5 or any(not (ROOT / name).is_file() for name in references):
        raise ValueError("README must link all five generated result images")
    for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8")):
        if not link.startswith(("https://", "http://", "#")) and not (ROOT / link.split("#", 1)[0]).exists():
            raise ValueError(f"Broken local README link: {link}")
    for name in references:
        image = plt.imread(ROOT / name)
        if image.shape[0] < 500 or image.shape[1] < 1000 or float(image.std()) < 0.01:
            raise ValueError(f"Empty or undersized image: {name}")
    paragraphs = (ROOT / "lab-submission" / "README.md").read_text(encoding="utf-8").strip().split("\n\n")
    if len(paragraphs) != 1:
        raise ValueError("Submission README must remain one paragraph")
    manifest = {"description": "Charts generated from retained results; input file hashes unchanged during rendering",
                "student_id": settings["student_id"], "input_files_sha256": original,
                "plot_files_sha256": {Path(name).name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in references}}
    (PLOTS / "plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Generated and checked five result images; source data unchanged; one-paragraph submission README preserved.")


if __name__ == "__main__":
    main()
