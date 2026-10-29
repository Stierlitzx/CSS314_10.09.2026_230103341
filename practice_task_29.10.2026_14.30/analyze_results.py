"""Verify raw samples and regenerate the requested results tables and report images."""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics
import sys

from benchmark_support import RESULTS, ROOT, STUDENT_ID, read_json, scaling_metrics, timestamp, write_json

NAMES = {"A": "Serial tail", "B": "Lock contention", "C": "Privatized state", "C-steady": "C, prestarted pool (supplementary)"}


def verify(records):
    for mode, record in records.items():
        assert record["student_id"] == STUDENT_ID
        assert len(record["samples"]) == 9 and record["runs_per_case"] == 3
        assert len({(row["P"], row["run"]) for row in record["samples"]}) == 9
        for p in (1, 2, 4):
            samples = [row for row in record["samples"] if row["P"] == p]
            assert len(samples) == 3 and all(row["time_ns"] > 0 and row["correct"] for row in samples)
            expected_median = statistics.median(row["time_ns"] for row in samples)
            metric = next(row for row in record["metrics"] if row["P"] == p)
            assert metric["median_time_ns"] == expected_median
            baseline = next(row["median_time_ns"] for row in record["metrics"] if row["P"] == 1)
            for name, value in scaling_metrics(baseline, expected_median, p).items():
                assert metric[name] is None if value is None else math.isclose(metric[name], value, rel_tol=1e-12, abs_tol=1e-12)
            for sample in samples:
                if mode == "A":
                    assert record["workload"] == {"total_chunks": 16, "chunk_size": 2_000_000,
                                                  "math_iterations": 31_999_984, "serial_pause_s": 0.4}
                    assert len(sample["chunk_values"]) == 16 and sample["actual_pause_ns"] >= 400_000_000
                    assert math.isclose(sample["checksum"], sample["reference_checksum"], abs_tol=1e-12)
                else:
                    total = 2_000_000 if mode == "B" else 20_000_000
                    assert record["workload"]["total_increments"] == total
                    assert sample["actual_total"] == sample["expected_total"] == total
                    assert sum(sample["assigned_increments"]) == total
                    if mode != "B":
                        assert sample["partial_totals"] == sample["assigned_increments"]
                    if mode == "C-steady":
                        assert len(set(sample["worker_pids"])) == p
                        assert sample["lifecycle_ns"] >= sample["time_ns"]


def metrics_table(records):
    lookup = {mode: {row["P"]: row for row in record["metrics"]} for mode, record in records.items()}
    lines = ["| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | S4 | E4 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("A", "B", "C"):
        m = lookup[mode]
        lines.append(f"| **{mode}: {NAMES[mode]}** | {m[1]['time_s']:.6f} | {m[2]['time_s']:.6f} | {m[4]['time_s']:.6f} | {m[2]['karp_flatt']:.6f} | {m[4]['karp_flatt']:.6f} | {m[4]['speedup']:.3f}× | {m[4]['efficiency_percent']:.2f}% |")
    return lookup, "\n".join(lines)


def reports(records, host, lookup, table):
    a, b, c, steady = (lookup[name] for name in ("A", "B", "C", "C-steady"))
    ratio = a[4]["karp_flatt"]/a[2]["karp_flatt"]
    flags = {"A_flatness_ratio_in_range": 0.8 <= ratio <= 1.25,
             "B_T4_greater_than_T1": b[4]["time_s"] > b[1]["time_s"],
             "B_e4_greater_than_e2": b[4]["karp_flatt"] > b[2]["karp_flatt"],
             "C_T4_less_than_T1": c[4]["time_s"] < c[1]["time_s"],
             "C_end_to_end_E4_over_70_percent": c[4]["efficiency_percent"] > 70,
             "C_supplementary_prestarted_E4_over_70_percent": steady[4]["efficiency_percent"] > 70}
    hardware = (f"**Student:** {STUDENT_ID} · **Session:** 29.10.2026, 14:30 · **Branch:** `practice-{STUDENT_ID}`\n\n"
                f"**Hardware:** {host['cpu_model']}; {host['physical_cores']} physical cores / {host['logical_threads']} logical threads; "
                f"{host['os']}; CPython {host['python'].split()[0]}; multiprocessing start method `{host['start_method']}`.\n\n")
    synthesis = (f"- **A — serial path:** e(4)/e(2) = **{ratio:.3f}**, within the required [0.80, 1.25] interval. "
                 "The absolute values include process startup/coordination, and are not guaranteed to match the handout's illustrative 0.10–0.14 range.\n"
                 f"- **B — contention:** T4/T1 = **{b[4]['time_s']/b[1]['time_s']:.2f}×**, so adding workers makes this implementation slower. "
                 f"Both e values are very large, but **e(4) < e(2)** in this run: the rubric's increasing-trend condition was not observed.\n"
                 f"- **C — privatization:** T4 < T1 and speedup is **{c[4]['speedup']:.2f}×**; "
                 f"end-to-end four-worker efficiency is **{c[4]['efficiency_percent']:.2f}%**, below the requested 70%. "
                 "The original pool-startup-inclusive measurements remain the primary results.\n")
    root_report = ROOT.parent / "RESULTS.md"
    marker = "<!-- practice-karp-flatt:230103341 -->"
    if root_report.exists() and marker not in root_report.read_text(encoding="utf-8"):
        raise RuntimeError("Repository-root RESULTS.md contains unrelated work; refusing to overwrite it")
    root_report.write_text(marker+"\n# Karp–Flatt Practice Results — 29.10.2026, 14:30\n\n"+hardware+
                           table+"\n\n"+synthesis+
                           f"\nThe separately measured prestarted C pool reaches **{steady[4]['speedup']:.2f}×** and **{steady[4]['efficiency_percent']:.2f}%** efficiency. "
                           "Its timing excludes startup/shutdown and is supplementary.\n\n"
                           "[Full report, images, raw samples and reproduction commands](practice_task_29.10.2026_14.30/README.md).\n", encoding="utf-8")
    report = "# Empirical Scalability Diagnostics — Results\n\n"+hardware+"## Required end-to-end measurements\n\n"+table+"\n\n"+synthesis
    report += "\n## Supplementary C: prestarted-pool diagnostic\n\n"
    report += "| P | Runtime (s) | Speedup | Efficiency | e(P) |\n|---:|---:|---:|---:|---:|\n"
    for p in (1, 2, 4):
        row = steady[p]
        e = "—" if p == 1 else f"{row['karp_flatt']:.6f}"
        report += f"| {p} | {row['time_s']:.6f} | {row['speedup']:.3f}× | {row['efficiency_percent']:.2f}% | {e} |\n"
    report += ("\nEvery worker is confirmed ready before this timer starts. The timer still includes task dispatch, private loops, IPC and the final reduction. "
               "Pool startup and shutdown are excluded; `lifecycle_ns` retains their full cost separately. "
               "The improvement supports startup/coordination overhead as a contributor; it does not isolate all hardware or scheduling effects. "
               "Even this diagnostic does not establish e≤0.03.\n\n## Raw timing samples\n\n"
               "Each cell is seconds. All trials are retained, with no outlier removal.\n\n"
               "| Mode | P | Run 1 | Run 2 | Run 3 | Median |\n|---|---:|---:|---:|---:|---:|\n")
    for mode, record in records.items():
        for p in (1, 2, 4):
            samples = sorted((row for row in record["samples"] if row["P"] == p), key=lambda row: row["run"])
            times = [row["time_ns"]/1e9 for row in samples]
            report += f"| {mode} | {p} | "+" | ".join(f"{t:.6f}" for t in times)+f" | {statistics.median(times):.6f} |\n"
    report += ("\n## Method and interpretation\n\n"
               "- A uses exactly 16 chunks, each summing `sin(i)*cos(i)` for i=1..1,999,999, preceded by `sleep(0.40)`. "
               f"The specified pause is only **{100*0.4/a[1]['time_s']:.2f}%** of this host's T1; the comment claiming ~12% is machine-dependent.\n"
               "- B performs exactly 2,000,000 increments under one explicit shared lock. The synchronized `Value('i', 0)` retains its default internal lock, as in the handout. "
               "The timings cover this combined synchronization cost, not isolated cache-coherence latency.\n"
               "- C performs exactly 20,000,000 private increments and reduces the results once. B and C therefore have **different total work**; compare scaling within each experiment rather than treating their raw runtime ratio as a same-work speedup.\n"
               "- Worker counts are 1, 2 and 4. Three trials rotate worker-count order to limit systematic order bias. Affinity and power settings are OS-managed. "
               "The baseline includes startup/shutdown, just like the supplied examples.\n"
               "- Partitions preserve remainders for optional non-divisible totals. All full-size counts and all A chunk results were checked outside timing.\n"
               "- Karp–Flatt is computed from **fixed total work** and each experiment's own T1. Values above one are retained; a diagnostic trend alone does not mathematically prove a unique hardware cause.\n"
               "- The handout's illustrative B timing table is numerically inconsistent with its formula: T1=0.65, T2=1.10 and T4=2.40 imply e(2)=2.385 and e(4)=4.590. "
               "This report applies the stated formula to the measured values.\n\n## Execution provenance\n\n")
    for mode, record in records.items():
        report += f"- **{mode}:** {record['started_at']} → {record['finished_at']}.\n"
    report += ("\nThe 29.10.2026 14:30 label is the scheduled session; captured timestamps are actual execution times. "
               "The folder was initially labelled 20.10 and renamed after the user's date correction; completed A measurements were preserved. "
               "An interrupted B attempt was not used in the recorded medians.\n\n"
               "See [README.md](README.md) for plots and commands. Repository-root [RESULTS.md](../RESULTS.md) contains the required submission summary.\n")
    (ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    return flags


def plots(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "plots"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white", "legend.frameon": False})
    colors = {"A": "#2563eb", "B": "#ea580c", "C": "#059669", "C-steady": "#7c3aed"}
    fig, (speedup, metric) = plt.subplots(1, 2, figsize=(12, 5.1))
    for mode, record in records.items():
        rows = record["metrics"]
        style = "--" if mode == "C-steady" else "-"
        speedup.plot([r["P"] for r in rows], [r["speedup"] for r in rows], "o"+style,
                     color=colors[mode], label=f"{mode}: {NAMES[mode]}", linewidth=2)
        metric.plot([r["P"] for r in rows if r["P"] > 1], [r["karp_flatt"] for r in rows if r["P"] > 1],
                    "o"+style, color=colors[mode], linewidth=2)
    speedup.plot([1, 2, 4], [1, 2, 4], ":", color="#64748b", label="Ideal linear scaling")
    speedup.set(ylabel="Speedup T1 / TP", title="Privatization recovers scaling", ylim=(0, 4.4))
    speedup.legend(fontsize=8, loc="upper left")
    metric.axhline(0, color="#94a3b8", linewidth=1)
    metric.set(ylabel="Karp-Flatt e(P)", title="Large contention overhead, without a monotonic surge")
    for ax in (speedup, metric):
        ax.set(xlabel="Worker processes P", xticks=[1, 2, 4] if ax is speedup else [2, 4])
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Karp-Flatt diagnostics | student 230103341", x=0.07, ha="left", weight="bold", fontsize=16)
    fig.text(0.07, 0.90, "End-to-end medians of three trials | Dashed purple series: separately timed prestarted C pool", color="#475569")
    fig.text(0.07, 0.025, "Source: results/experiment_*.json | Work is fixed within each mode; each mode uses its own single-worker baseline.", fontsize=9)
    fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.27)
    fig.savefig(folder / "scaling_diagnostics.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9))
    for mode, ax in zip(("A", "B", "C"), axes):
        record = records[mode]
        values = [row["time_s"] for row in record["metrics"]]
        bars = ax.bar([1, 2, 3], values, color=colors[mode], alpha=0.85, width=0.55)
        ax.bar_label(bars, fmt="%.3f s", padding=5, fontsize=9)
        for index, p in enumerate((1, 2, 4), 1):
            raw = [row["time_ns"]/1e9 for row in record["samples"] if row["P"] == p]
            ax.scatter([index-0.07, index, index+0.07], raw, color="#0f172a", s=15, zorder=4)
        max_raw = max(row["time_ns"]/1e9 for row in record["samples"])
        ax.set(title=f"{mode}: {NAMES[mode]}", xticks=[1, 2, 3], xticklabels=[1, 2, 4],
               xlabel="Worker processes P", ylabel="Elapsed seconds", ylim=(0, max_raw*1.27))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Recorded end-to-end runtimes", x=0.07, ha="left", weight="bold", fontsize=16)
    fig.text(0.07, 0.90, "A: 16 math chunks + 0.40 s pause | B: 2 million locked increments | C: 20 million private increments", color="#475569")
    fig.text(0.07, 0.025, "Bars: medians; dots: all raw trials. Panels use different vertical scales and different workloads.", fontsize=9)
    fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.32)
    fig.savefig(folder / "runtimes.png", dpi=170)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--plots", action="store_true", help="Also render PNGs; requires optional matplotlib")
    args = cli.parse_args()
    paths = {mode: RESULTS / f"experiment_{mode.lower()}.json" for mode in NAMES}
    original = {mode: hashlib.sha256(path.read_bytes()).hexdigest() for mode, path in paths.items()}
    records = {mode: read_json(path) for mode, path in paths.items()}
    verify(records)
    lookup, table = metrics_table(records)
    flags = reports(records, read_json(RESULTS / "host_profile.json"), lookup, table)
    with (RESULTS / "scaling_metrics.tsv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["experiment", "P", "median_time_ns", "time_s", "speedup", "efficiency_percent", "karp_flatt"]
        writer = csv.DictWriter(stream, columns, delimiter="\t")
        writer.writeheader()
        for mode, record in records.items():
            writer.writerows({"experiment": mode, **row} for row in record["metrics"])
    if args.plots:
        plots(records)
    assert original == {mode: hashlib.sha256(path.read_bytes()).hexdigest() for mode, path in paths.items()}
    write_json(RESULTS / "verification.json", {"status": "PASS", "checked_at": timestamp(),
               "verified_samples": 36, "correctness_scope": "Recorded checksums, exact counts, partitions, raw timing medians and scaling formulas",
               "rubric_observations": flags, "raw_data_sha256": original})
    print(table)
    print("Numerical/metric verification: PASS (36 samples). Rubric observations:", flags)


if __name__ == "__main__":
    main()
