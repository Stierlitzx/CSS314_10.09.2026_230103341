import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
N = 10_341_000
PHASE_FILES = [
    "results_seq.csv",
    "results_scale.csv",
    "results_scale6.csv",
    "results_false_sharing.csv",
    "results_scheduling.csv",
]
FIELDNAMES = [
    "phase", "variant", "schedule", "chunk", "threads", "run",
    "seconds", "max_steps", "checksum", "hits_over_100",
]

rows = []
for name in PHASE_FILES:
    with (ROOT / "raw" / name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["chunk"] = int(row["chunk"])
            row["threads"] = int(row["threads"])
            row["run"] = int(row["run"])
            row["seconds"] = float(row["seconds"])
            row["max_steps"] = int(row["max_steps"])
            row["checksum"] = int(row["checksum"])
            row["hits_over_100"] = int(row["hits_over_100"])
            rows.append(row)

with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

checksums = {row["checksum"] for row in rows}
maxima = {row["max_steps"] for row in rows}
hit_counts = {row["hits_over_100"] for row in rows}
if len(checksums) != 1 or len(maxima) != 1 or len(hit_counts) != 1:
    raise SystemExit("Validation failed: benchmark results are inconsistent")


def group_rows(*, phase, variant=None, threads=None, schedule=None):
    selected = [
        row for row in rows
        if row["phase"] == phase
        and (variant is None or row["variant"] == variant)
        and (threads is None or row["threads"] == threads)
        and (schedule is None or row["schedule"] == schedule)
    ]
    return sorted(selected, key=lambda row: row["run"])


def measured_average(group):
    measured = [row["seconds"] for row in group if row["run"] >= 2]
    if len(measured) != 2:
        raise ValueError("Each configuration must contain measured Runs 2 and 3")
    return sum(measured) / 2.0


seq_group = group_rows(phase="sequential", variant="sequential", threads=1)
t_seq = measured_average(seq_group)

scale_groups = {}
for threads in sorted({row["threads"] for row in rows if row["phase"] == "scaling"}):
    scale_groups[threads] = group_rows(phase="scaling", threads=threads)

scale_avg = {k: measured_average(v) for k, v in scale_groups.items()}
s_emp = {k: t_seq / avg for k, avg in scale_avg.items()}
s2 = s_emp[2]
p = 2.0 * (1.0 - 1.0 / s2)
s_theo = {k: 1.0 / ((1.0 - p) + p / k) for k in scale_avg}
delta = {k: s_theo[k] - s_emp[k] for k in scale_avg}

false_variants = sorted({row["variant"] for row in rows if row["phase"] == "false_sharing"})
false_groups = {
    variant: group_rows(phase="false_sharing", variant=variant)
    for variant in false_variants
}
false_avg = {variant: measured_average(group) for variant, group in false_groups.items()}
false_throughput = {variant: N / avg for variant, avg in false_avg.items()}
false_penalty = false_avg["naive_shared_line"] / false_avg["cache_padded"]

schedule_names = ["static", "static_1000", "dynamic_100", "dynamic_10000", "guided"]
schedule_groups = {
    name: group_rows(phase="scheduling", schedule=name)
    for name in schedule_names
}
schedule_avg = {name: measured_average(group) for name, group in schedule_groups.items()}
schedule_throughput = {name: N / avg for name, avg in schedule_avg.items()}

summary_rows = []
for threads, group in scale_groups.items():
    run_times = {row["run"]: row["seconds"] for row in group}
    summary_rows.append({
        "table": "scaling", "configuration": f"k={threads}", "threads": threads,
        "run1": run_times[1], "run2": run_times[2], "run3": run_times[3],
        "average": scale_avg[threads], "speedup_empirical": s_emp[threads],
        "speedup_theoretical": s_theo[threads], "reality_gap": delta[threads],
        "throughput_iter_per_s": N / scale_avg[threads],
    })
for variant, group in false_groups.items():
    run_times = {row["run"]: row["seconds"] for row in group}
    summary_rows.append({
        "table": "false_sharing", "configuration": variant,
        "threads": group[0]["threads"], "run1": run_times[1],
        "run2": run_times[2], "run3": run_times[3], "average": false_avg[variant],
        "speedup_empirical": "", "speedup_theoretical": "", "reality_gap": "",
        "throughput_iter_per_s": false_throughput[variant],
    })
for name, group in schedule_groups.items():
    run_times = {row["run"]: row["seconds"] for row in group}
    summary_rows.append({
        "table": "scheduling", "configuration": name,
        "threads": group[0]["threads"], "run1": run_times[1],
        "run2": run_times[2], "run3": run_times[3], "average": schedule_avg[name],
        "speedup_empirical": t_seq / schedule_avg[name],
        "speedup_theoretical": "", "reality_gap": "",
        "throughput_iter_per_s": schedule_throughput[name],
    })

with (ROOT / "results_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

ks = list(scale_avg.keys())
plt.figure(figsize=(10, 6), dpi=180)
plt.plot(ks, [s_emp[k] for k in ks], "o-", linewidth=2, label="Empirical speedup")
plt.plot(ks, [s_theo[k] for k in ks], "s--", linewidth=2, label=f"Amdahl fit (p={p:.6f})")
plt.plot(ks, ks, ":", linewidth=2, label="Linear ideal")
plt.xlabel("OpenMP threads (k)")
plt.ylabel("Speedup relative to sequential baseline")
plt.title("Collatz OpenMP Scaling: Amdahl Reality Gap")
plt.xticks(ks)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "speedup_plot.png")
plt.close()

lines = [
    "CSS314 Collatz benchmark summary",
    f"N: {N}", f"Checksum: {checksums.pop()}",
    f"Maximum stopping time: {maxima.pop()}",
    f"Numbers with >100 steps: {hit_counts.pop()}",
    f"Sequential T_seq: {t_seq:.6f} s",
    f"Derived S_emp(2): {s2:.6f}",
    f"Derived parallel fraction p: {p:.6f}", "", "Scaling:",
]
for k in ks:
    g = {row["run"]: row["seconds"] for row in scale_groups[k]}
    lines.append(
        f"k={k:2d}: runs={g[1]:.6f}/{g[2]:.6f}/{g[3]:.6f} s, "
        f"avg={scale_avg[k]:.6f} s, S_emp={s_emp[k]:.4f}, "
        f"S_theo={s_theo[k]:.4f}, delta={delta[k]:+.4f}"
    )
lines.extend(["", "False sharing:"])
for variant in false_variants:
    g = {row["run"]: row["seconds"] for row in false_groups[variant]}
    lines.append(
        f"{variant}: runs={g[1]:.6f}/{g[2]:.6f}/{g[3]:.6f} s, "
        f"avg={false_avg[variant]:.6f} s, throughput={false_throughput[variant]:,.0f} iter/s"
    )
lines.append(f"Naive/padded penalty ratio: {false_penalty:.6f}")
lines.extend(["", "Scheduling:"])
for name in schedule_names:
    g = {row["run"]: row["seconds"] for row in schedule_groups[name]}
    lines.append(
        f"{name}: runs={g[1]:.6f}/{g[2]:.6f}/{g[3]:.6f} s, "
        f"avg={schedule_avg[name]:.6f} s, throughput={schedule_throughput[name]:,.0f} iter/s, "
        f"speedup_vs_seq={t_seq / schedule_avg[name]:.4f}"
    )

summary_text = "\n".join(lines) + "\n"
(ROOT / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")
print(summary_text)
print("Wrote results.csv, results_summary.csv, speedup_plot.png, and analysis_summary.txt")
