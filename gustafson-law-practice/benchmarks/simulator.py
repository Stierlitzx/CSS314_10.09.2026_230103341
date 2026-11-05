"""Task 21: configurable CSV simulator and strong/weak scaling overlays."""
import argparse
import csv
from pathlib import Path

from models import amdahl, gustafson

ROOT = Path(__file__).resolve().parents[1]


def parse_cores(value):
    try:
        cores = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers") from error
    if not cores or not all(1 <= p <= 65536 for p in cores):
        raise argparse.ArgumentTypeError("core counts must be in [1, 65536]")
    return cores


def parse_fractions(value):
    import math
    try:
        fractions = sorted(set(float(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated fractions") from error
    if not fractions or not all(math.isfinite(a) and 0.001 <= a <= 0.20 for a in fractions):
        raise argparse.ArgumentTypeError("fractions must be in [0.001, 0.20]")
    return fractions


def simulate(cores, fractions, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["P", "fraction", "amdahl_speedup", "gustafson_speedup",
                  "amdahl_efficiency", "gustafson_efficiency"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rows = []
        for fraction in fractions:
            for p in cores:
                sa, sg = amdahl(p, fraction), gustafson(p, fraction)
                row = dict(P=p, fraction=fraction, amdahl_speedup=sa,
                           gustafson_speedup=sg, amdahl_efficiency=sa / p,
                           gustafson_efficiency=sg / p)
                writer.writerow(row)
                rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    fractions = sorted({r["fraction"] for r in rows})
    for a in fractions:
        subset = [r for r in rows if r["fraction"] == a]
        x = [r["P"] for r in subset]
        line, = axes[0].plot(x, [r["gustafson_speedup"] for r in subset],
                             label=f"Gustafson, alpha={a:g}")
        axes[0].plot(x, [r["amdahl_speedup"] for r in subset], "--", color=line.get_color(),
                     label=f"Amdahl, s={a:g}")
        axes[1].plot(x, [r["gustafson_efficiency"] for r in subset], color=line.get_color())
        axes[1].plot(x, [r["amdahl_efficiency"] for r in subset], "--", color=line.get_color())
    x = sorted({r["P"] for r in rows})
    axes[0].plot(x, x, ":", color="black", label="Ideal S=P")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Processors P")
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Speedup (log scale)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Parallel efficiency")
    axes[1].set_ylim(0, 1.03)
    fig.suptitle("Fixed-size and scaled-size predictions: equal numerical fractions, different workloads")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=parse_cores,
                        default=parse_cores(",".join(str(2 ** i) for i in range(17))))
    parser.add_argument("--fractions", type=parse_fractions, default=parse_fractions("0.001,0.04,0.08,0.20"))
    parser.add_argument("--csv", type=Path, default=ROOT / "calculations" / "simulated_scaling.csv")
    parser.add_argument("--plot", type=Path, default=ROOT / "plots" / "model_overlay")
    parser.add_argument("--no-plots", action="store_true", help="CSV-only mode needs no third-party packages")
    args = parser.parse_args()
    rows = simulate(args.cores, args.fractions, args.csv)
    if not args.no_plots:
        plot(rows, args.plot)
    print(f"Exported {len(rows)} predictions to {args.csv.resolve()}")


if __name__ == "__main__":
    main()