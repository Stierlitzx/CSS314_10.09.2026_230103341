"""Render recorded CUDA measurements and extract the original notebook image."""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


def main():
    folder = Path(__file__).resolve().parent
    results = json.loads((folder / "CUDA_Results_230103341.json").read_text(encoding="utf-8"))
    notebook = json.loads((folder / "CUDA_Lab01_230103341.ipynb").read_text(encoding="utf-8"))
    plots = folder / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    rows = results["benchmark"]
    sizes = [row["N"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, label, color in (
        ("cpu_ms", "CPU: pure Python", "#334155"),
        ("gpu_total_ms", "GPU: total roundtrip", "#2563eb"),
        ("gpu_kernel_ms", "GPU: preloaded + launch/sync", "#d97706"),
    ):
        ax.loglog(sizes, [row[key] for row in rows], "o-", label=label, color=color, linewidth=2)
    crossover = results["crossover_N"]
    if crossover is not None:
        ax.axvline(crossover, color="#15803d", linestyle="--", label=f"First tested crossover: {crossover:,}")
    ax.set(xlabel="Array size N (float32 elements)", ylabel="Elapsed time (ms, logarithmic scale)",
           title="Vector doubling: recorded CPU and GPU timings")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots / "vector_doubling.png", dpi=180)
    plt.close(fig)

    geometry = results["thread_geometry"]
    threads_per_block = geometry["launched"] // geometry["blocks"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for global_id in range(geometry["launched"]):
        block, thread = divmod(global_id, threads_per_block)
        active = global_id < geometry["active"]
        ax.add_patch(Rectangle((thread - 0.45, block - 0.38), 0.9, 0.76,
                               facecolor="#2563eb" if active else "#d97706"))
        ax.text(thread, block, str(global_id), ha="center", va="center", color="white", weight="bold")
    ax.set(xlim=(-0.6, threads_per_block - 0.4), ylim=(geometry["blocks"] - 0.5, -0.6),
           xticks=range(threads_per_block), yticks=range(geometry["blocks"]),
           xlabel="Local thread ID", ylabel="Block ID",
           title=f"Recorded thread geometry: {geometry['active']} active / {geometry['idle']} guarded\nCell labels are global thread IDs")
    ax.legend(handles=[Patch(color="#2563eb", label="Active"), Patch(color="#d97706", label="Guarded")],
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "thread_geometry.png", dpi=180)
    plt.close(fig)

    fig, (correctness, timing) = plt.subplots(1, 2, figsize=(11, 4.8))
    trials = results["naive_trials"]
    trial_ids = [row["trial"] for row in trials]
    correctness.plot(trial_ids, [row["output"] for row in trials], "o-", color="#d97706", label="Naive")
    correctness.plot(trial_ids, [row[0] for row in results["atomic_trials"]], "s-", color="#2563eb", label="Atomic")
    correctness.axhline(trials[0]["target"], color="#334155", linestyle="--", label="Target: 50,000")
    correctness.set(yscale="symlog", ylim=(0, 150000), xticks=trial_ids,
                    xlabel="Trial", ylabel="Calculated sum (symlog scale)", title="Correctness: 50,000 ones")
    correctness.grid(True, alpha=0.2)
    correctness.legend(loc="center right")
    bars = timing.bar(["Naive", "Atomic"], [results["naive_median_ms"], results["atomic_median_ms"]],
                      color=["#d97706", "#2563eb"], width=0.6)
    timing.bar_label(bars, fmt="%.4f ms", padding=5)
    timing.set(ylabel="Median elapsed time (ms)", title="Five timed trials per kernel",
               ylim=(0, max(results["naive_median_ms"], results["atomic_median_ms"]) * 1.25))
    timing.grid(axis="y", alpha=0.2)
    timing.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(plots / "reduction.png", dpi=180)
    plt.close(fig)

    # Export the actual rendered GPU result instead of recomputing the vignette.
    images = [output["data"]["image/png"]
              for cell in notebook["cells"] if cell["cell_type"] == "code"
              and "radial_vignette_2d" in "".join(cell["source"])
              for output in cell["outputs"] if "image/png" in output.get("data", {})]
    if len(images) != 1:
        raise ValueError(f"Expected one retained vignette PNG, found {len(images)}")
    image = "".join(images[0]) if isinstance(images[0], list) else images[0]
    (plots / "radial_vignette.png").write_bytes(base64.b64decode(image))
    print(f"Generated four report images in {plots}")


if __name__ == "__main__":
    main()
