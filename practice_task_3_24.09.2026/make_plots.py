"""Generate all lab plots from the raw CSV results."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def read_text(path):
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def read_csv_rows(path):
    rows = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


# ---------------- Lab 1: oversubscription sweep + saturation ----------------
rows = list(csv.DictReader(read_csv_rows(ROOT / "lab1/results/task1_2_oversubscription.csv")))
p1 = [int(r["threads"]) for r in rows]
avg = [float(r["avg_ms"]) for r in rows]

rows = list(csv.DictReader(read_csv_rows(ROOT / "lab1/results/task1_3_saturation.csv")))
p2 = [int(r["threads"]) for r in rows]
thr = [float(r["throughput_Msqrt_per_s"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(p1, avg, "o-", color="tab:red")
ax1.set_xlabel("Team size P"); ax1.set_ylabel("Fork/join time (ms)")
ax1.set_title("Lab 1 - Thread team fork/join cost")
ax1.axvline(12, ls="--", c="gray", lw=1)
ax1.text(12.3, min(avg), "12 logical CPUs", rotation=90, fontsize=8)
ax1.grid(alpha=0.3)
ax2.plot(p2, thr, "s-", color="tab:green")
ax2.set_xlabel("Team size P"); ax2.set_ylabel("Throughput (Msqrt/s)")
ax2.set_title("Lab 1 - CPU saturation (10M sqrt/thread)")
ax2.axvline(6, ls="--", c="gray", lw=1)
ax2.text(6.3, min(thr), "6 physical cores", rotation=90, fontsize=8)
ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(PLOTS / "lab1_forkjoin.png", dpi=150); plt.close(fig)

# ---------------- Lab 2: speedup vs ideal ----------------
txt = read_text(ROOT / "lab2/results/lab2_all_output.txt")
scale_section = txt.split("Task 2.3/2.4")[1]
rows = list(csv.DictReader([l for l in scale_section.splitlines()
                            if l.strip() and (l.startswith("threads,") or l[0].isdigit())]))
p = [int(r["threads"]) for r in rows]
speedup = [float(r["speedup"]) for r in rows]
eff = [float(r["efficiency"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(p, speedup, "o-", label="Measured speedup S(P)", color="tab:blue")
ax1.plot(p, p, "--", label="Ideal linear speedup", color="tab:gray")
ax1.set_xlabel("Threads P"); ax1.set_ylabel("Speedup S(P) = T(1)/T(P)")
ax1.set_xticks(p)
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(p, eff, "s--", label="Efficiency E(P)", color="tab:orange")
ax2.set_ylabel("Parallel efficiency E(P) = S(P)/P"); ax2.set_ylim(0, 1.15)
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
ax1.set_title("Lab 2 - Pi reduction strong scaling (N = 100M)")
fig.tight_layout(); fig.savefig(PLOTS / "lab2_speedup.png", dpi=150); plt.close(fig)


# ---------------- Lab 3: heatmap threads x chunk + static ----------------
rows = [r for r in csv.DictReader(read_csv_rows(ROOT / "lab3/results/task3_matrix.csv"))
        if r["threads"].isdigit()]
threads = [2, 4, 8, 16]
chunks = [1, 16, 64, 256]
dyn = np.zeros((len(threads), len(chunks)))
static_avg = {}
for r in rows:
    t, c, ms = int(r["threads"]), int(r["chunk"]), float(r["time_ms"])
    if r["scheduler"] == "dynamic":
        dyn[threads.index(t), chunks.index(c)] += ms / 3.0
    else:
        static_avg.setdefault(t, []).append(ms)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
im = ax1.imshow(dyn, cmap="viridis_r", aspect="auto")
ax1.set_xticks(range(len(chunks)), chunks)
ax1.set_yticks(range(len(threads)), threads)
ax1.set_xlabel("Chunk size C"); ax1.set_ylabel("Threads P")
ax1.set_title("Lab 3 - Dynamic scheduling mean time (ms)")
for i in range(len(threads)):
    for j in range(len(chunks)):
        ax1.text(j, i, f"{dyn[i, j]:.0f}", ha="center", va="center",
                 color="white" if dyn[i, j] > dyn.max() * 0.55 else "black", fontsize=9)
fig.colorbar(im, ax=ax1, label="ms")
x = np.arange(len(threads))
ax2.bar(x - 0.2, [np.mean(static_avg[t]) for t in threads], 0.4, label="Static (block)")
best_dyn = [dyn[i].min() for i in range(len(threads))]
ax2.bar(x + 0.2, best_dyn, 0.4, label="Dynamic (best chunk)")
ax2.set_xticks(x, threads); ax2.set_xlabel("Threads P"); ax2.set_ylabel("Mean time (ms)")
ax2.set_title("Lab 3 - Static vs. best dynamic")
ax2.legend(); ax2.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(PLOTS / "lab3_scheduling.png", dpi=150); plt.close(fig)

# ---------------- Lab 4: false sharing scaling ----------------
rows = list(csv.DictReader(read_csv_rows(ROOT / "lab4/results/task4_scaling.csv")))
data = {}
for r in rows:
    data.setdefault(r["variant"], {}).setdefault(int(r["threads"]), []).append(float(r["time_ms"]))

fig, ax = plt.subplots(figsize=(7, 5))
for variant, style in [("unpadded", "o-"), ("padded", "s-"), ("local", "^-")]:
    ps = sorted(data[variant])
    means = [np.mean(data[variant][pp]) for pp in ps]
    ax.plot(ps, means, style, label={"unpadded": "Unpadded (false sharing)",
                                     "padded": "Padded (64B stride)",
                                     "local": "Thread-local register"}[variant])
ax.set_xlabel("Threads P"); ax.set_ylabel("Time (ms)")
ax.set_title("Lab 4 - False sharing scaling (100M increments/thread)")
ax.set_xticks(sorted(data["unpadded"]))
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(PLOTS / "lab4_false_sharing.png", dpi=150); plt.close(fig)

# ---------------- Lab 5: cutoff sweep ----------------
txt = read_text(ROOT / "lab5/results/task5_results.txt")
cut_rows = list(csv.DictReader([l for l in txt.splitlines()
                                if l.strip() and (l.startswith("cutoff,") or l[0].isdigit())]))
k = [int(r["cutoff"]) for r in cut_rows]
t = [float(r["avg_time_ms"]) for r in cut_rows]

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(k, t, "o-", color="tab:purple")
best = int(np.argmin(t))
ax.annotate(f"optimum K={k[best]} ({t[best]:.0f} ms)", xy=(k[best], t[best]),
            xytext=(k[best] / 30, t[best] + 15), arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Sequential cutoff K (log scale)"); ax.set_ylabel("Avg time (ms)")
ax.set_title("Lab 5 - Parallel merge sort cutoff sweep (N = 5M)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(PLOTS / "lab5_cutoff.png", dpi=150); plt.close(fig)

print("Plots written to", PLOTS)
