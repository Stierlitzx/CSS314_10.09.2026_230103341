"""Tasks 23-24: validate raw measurements, fit alpha, export statistics and plots."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from models import amdahl, gustafson

ROOT = Path(__file__).resolve().parents[1]


def load_results(path, minimum_runs=5, minimum_counts=5):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Dataset is empty")
    groups = defaultdict(list)
    seen, ks, run_ids = set(), set(), set()
    for raw in rows:
        p, run, k = int(raw["P"]), int(raw["run"]), int(raw["k"])
        ts, tp = float(raw["serial_seconds"]), float(raw["parallel_seconds"])
        if min(p, run, k) < 1 or not all(math.isfinite(t) and t > 0 for t in (ts, tp)):
            raise ValueError("Invalid processor count, run, k, or timing")
        if (p, run) in seen:
            raise ValueError("Duplicate processor/run pair")
        seen.add((p, run))
        ks.add(k)
        run_ids.add(raw["run_id"])
        if int(raw["total_points"]) != p * k or raw["correct"] != "True":
            raise ValueError("Incorrect weak-scaling workload or failed checksum")
        pids = [int(pid) for pid in raw["worker_pids"].split(";")]
        if len(pids) != p or len(set(pids)) != p or not all(pid > 0 for pid in pids):
            raise ValueError("Expected exactly P distinct live-process identities per sample")
        serial, parallel = float(raw["serial_integral"]), float(raw["parallel_integral"])
        if not math.isclose(serial, parallel, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Serial/parallel quadrature mismatch")
        error = abs(parallel - math.pi)
        if error > max(1e-10, 2 / (p * k) ** 2):
            raise ValueError("Quadrature outside conservative error tolerance")
        if not math.isclose(float(raw["absolute_pi_error"]), error, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError("Stored quadrature error inconsistent with measurement")
        speedup = ts / tp
        if not math.isclose(speedup, float(raw["scaled_speedup"]), rel_tol=1e-12):
            raise ValueError("Stored speedup inconsistent with measured timings")
        groups[p].append(dict(run=run, serial_seconds=ts, parallel_seconds=tp, speedup=speedup))
    if len(ks) != 1 or len(run_ids) != 1:
        raise ValueError("Mixed workload sizes or benchmark sessions")
    if 1 not in groups or len(groups) < minimum_counts:
        raise ValueError("Need P=1 and at least five distinct processor counts")
    expected_runs = set(range(1, len(groups[1]) + 1))
    if len(expected_runs) < minimum_runs:
        raise ValueError("Need at least five independent trials per configuration")
    summary = []
    t1 = statistics.mean(r["parallel_seconds"] for r in groups[1])
    for p, samples in sorted(groups.items()):
        if {r["run"] for r in samples} != expected_runs:
            raise ValueError("Every configuration must have the same contiguous trial numbers")
        n = len(samples)
        mean_t = statistics.mean(r["parallel_seconds"] for r in samples)
        speeds = [r["speedup"] for r in samples]
        speed_sd = statistics.stdev(speeds) if n > 1 else 0.0
        summary.append(dict(P=p, runs=n, k=next(iter(ks)),
                            mean_parallel_seconds=mean_t,
                            sd_parallel_seconds=statistics.stdev(r["parallel_seconds"] for r in samples) if n > 1 else 0.0,
                            mean_serial_seconds=statistics.mean(r["serial_seconds"] for r in samples),
                            mean_scaled_speedup=statistics.mean(speeds), sd_scaled_speedup=speed_sd,
                            se_scaled_speedup=speed_sd / math.sqrt(n),
                            mean_scaled_efficiency=statistics.mean(speeds) / p,
                            weak_runtime_efficiency=t1 / mean_t))
    return summary


def fit(summary):
    """Unweighted OLS over configuration means, with S=P-alpha(P-1)."""
    if len(summary) < 2:
        raise ValueError("Need at least two processor counts")
    counts = set()
    for row in summary:
        p = row["P"]
        if isinstance(p, bool) or not isinstance(p, int) or p < 1:
            raise ValueError("Processor counts must be positive integers")
        if p in counts:
            raise ValueError("Need unique processor counts")
        counts.add(p)
        if not math.isfinite(row["mean_scaled_speedup"]):
            raise ValueError("Mean speedups must be finite")
    denominator = sum((r["P"] - 1) ** 2 for r in summary)
    if denominator == 0:
        raise ValueError("Alpha unidentifiable from P=1 alone")
    alpha = sum((r["P"] - 1) * (r["P"] - r["mean_scaled_speedup"]) for r in summary) / denominator
    residuals = [r["mean_scaled_speedup"] - gustafson(r["P"], alpha)
                 if 0 <= alpha <= 1 else r["mean_scaled_speedup"] - (r["P"] - alpha * (r["P"] - 1))
                 for r in summary]
    sse = sum(value ** 2 for value in residuals)
    mean_s = statistics.mean(r["mean_scaled_speedup"] for r in summary)
    sst = sum((r["mean_scaled_speedup"] - mean_s) ** 2 for r in summary)
    alpha_se = math.sqrt(sse / (len(summary) - 1) / denominator)
    diagnostic = statistics.linear_regression([r["P"] for r in summary],
                                            [r["mean_scaled_speedup"] for r in summary])
    return dict(alpha_observed=alpha, alpha_in_physical_range=0 <= alpha <= 1,
                fitting_method="unweighted least squares of per-configuration mean paired Ts(P*k)/Tp(P*k); intercept fixed at S(1)=1",
                formula="alpha=sum((P-1)*(P-Smean))/sum((P-1)^2)",
                alpha_residual_standard_error=alpha_se,
                alpha_se_caveat="OLS residual-based diagnostic, not a direct measurement or calibrated confidence interval; heteroscedasticity/model mismatch possible",
                rmse=math.sqrt(sse / len(summary)), r_squared=1 - sse / sst if sst else None,
                free_intercept_diagnostic=dict(intercept=diagnostic.intercept, slope=diagnostic.slope,
                                               expected_intercept=alpha, expected_slope=1-alpha),
                residuals=[dict(P=r["P"], residual=value) for r, value in zip(summary, residuals)],
                warnings=["P=1 startup-inclusive speedup need not be one: reference is direct, not a spawned worker.",
                          "A fitted effective alpha includes overhead and imbalance; it is not instrumentation of serial code.",
                          "Amdahl with the same numerical fraction is a different-workload illustration, not a matched-work prediction."])


def save_summary(summary, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def plot(summary, fitted, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, (ax, runtime) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    max_p = max(r["P"] for r in summary)
    dense = [1 + (max_p - 1) * i / 300 for i in range(301)]
    alpha = fitted["alpha_observed"]
    ax.plot(dense, dense, "--", color="0.45", label="Ideal S=P")
    ax.plot(dense, [p - alpha * (p - 1) for p in dense], color="#0072B2",
            label=f"Gustafson fit: alpha={alpha:.4f}")
    if fitted["alpha_in_physical_range"]:
        ax.plot(dense, [1 / (alpha + (1-alpha)/p) for p in dense], color="#D55E00",
                label="Amdahl: same numeric fraction*")
    ax.errorbar([r["P"] for r in summary], [r["mean_scaled_speedup"] for r in summary],
                yerr=[r["se_scaled_speedup"] for r in summary], fmt="o", markersize=4,
                color="#009E73", capsize=4, label=f"Measured mean ± 1 SE ({summary[0]['runs']} trials)")
    ax.set(xlabel="Worker processes P", ylabel="Matched-work scaled speedup Ts(Pk) / Tp(Pk)",
           title=f"Measured vs. model (fit RMSE={fitted['rmse']:.3f})")
    ax.legend(fontsize=8)
    runtime.errorbar([r["P"] for r in summary], [r["mean_parallel_seconds"] for r in summary],
                     yerr=[r["sd_parallel_seconds"] / math.sqrt(r["runs"]) for r in summary],
                     fmt="o-", capsize=4, color="#CC79A7", label="End-to-end runtime ± 1 SE")
    runtime.axhline(summary[0]["mean_parallel_seconds"], linestyle="--", color="0.45",
                    label="Fixed-time ideal: T1")
    runtime.set(xlabel="Worker processes P", ylabel="Wall-clock seconds",
                title=f"Weak scaling: k={summary[0]['k']:,} evaluations / worker")
    runtime.legend(fontsize=8)
    for axis in (ax, runtime):
        axis.grid(alpha=0.2)
        axis.set_xlim(0.7, max_p + 0.3)
        axis.set_ylim(bottom=0)
    fig.suptitle("Gustafson practice • 230103341 • real process measurements")
    fig.supxlabel("*Amdahl curve uses a different fraction reference/workload; SE bars are sampling precision, not model uncertainty.", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks" / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--plot", type=Path, default=ROOT / "plots" / "scaling_comparison")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = load_results(args.input)
    fitted = fit(summary)
    fitted.update(source=str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
                  configurations=len(summary), trials=sum(r["runs"] for r in summary))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(summary, args.output_dir / "summary.csv")
    (args.output_dir / "fit.json").write_text(json.dumps(fitted, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_plots:
        plot(summary, fitted, args.plot)
    print(json.dumps(fitted, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()