"""Standard-library timing, topology and fixed-work scaling helpers."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STUDENT_ID = "230103341"
WORKERS = (1, 2, 4)


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def partition(total, workers):
    if total < 0 or workers < 1:
        raise ValueError("total must be nonnegative and workers must be positive")
    quotient, remainder = divmod(total, workers)
    return [quotient + (i < remainder) for i in range(workers)]


def scaling_metrics(t1, tp, workers):
    if t1 <= 0 or tp <= 0 or workers < 1:
        raise ValueError("Timings and worker count must be positive")
    speedup = t1 / tp
    return {"speedup": speedup, "efficiency_percent": 100 * speedup / workers,
            "karp_flatt": (workers * tp / t1 - 1) / (workers - 1) if workers > 1 else None}


def capture_host():
    host = {"captured_at": timestamp(), "student_id": STUDENT_ID,
            "scheduled_practice": "2026-10-29 14:30",
            "cpu_model": platform.processor(), "physical_cores": None,
            "logical_threads": os.cpu_count(), "os": platform.platform(),
            "python": sys.version, "python_executable": sys.executable,
            "start_method": "spawn", "affinity": "OS-managed; no affinity or power-plan changes",
            "timer": {"name": "perf_counter_ns", "resolution_seconds": time.get_clock_info("perf_counter").resolution}}
    if os.name == "nt":
        command = "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json"
        try:
            info = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True, timeout=30))
            packages = info if isinstance(info, list) else [info]
            host.update(cpu_model="; ".join(p["Name"].strip() for p in packages),
                        physical_cores=sum(p["NumberOfCores"] for p in packages),
                        topology_source="Windows CIM Win32_Processor", cpu_packages=packages)
        except (subprocess.SubprocessError, ValueError, KeyError) as error:
            host["topology_error"] = str(error)
    write_json(RESULTS / "host_profile.json", host)
    print(f"Execution Cores: {host['logical_threads']} | Physical cores: {host['physical_cores']}", flush=True)
    return host


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--runs", type=positive, default=3)
    return result


def run_suite(experiment, measure, workload, runs):
    RESULTS.mkdir(exist_ok=True)
    if not (RESULTS / "host_profile.json").exists():
        capture_host()
    if (os.cpu_count() or 1) < max(WORKERS):
        raise RuntimeError("This handout requires four available logical execution cores")
    started = timestamp()
    samples = []
    path = RESULTS / f"experiment_{experiment.lower()}.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("status") == "IN_PROGRESS" and previous["workload"] == workload and previous["runs_per_case"] == runs:
            samples = previous["samples"]
            started = previous["started_at"]
            print(f"Resuming {experiment}: {len(samples)} completed samples retained", flush=True)
    record = {"experiment": experiment, "student_id": STUDENT_ID, "status": "IN_PROGRESS",
              "started_at": started, "runs_per_case": runs, "workload": workload,
              "timing_scope": "End-to-end, including process/pool startup, IPC and shutdown; validation excluded",
              "samples": samples}
    for repeat in range(runs):
        # Rotate the order once per repetition; retain every sample, including outliers.
        order = WORKERS[repeat % len(WORKERS):] + WORKERS[:repeat % len(WORKERS)]
        for workers in order:
            if any(row["P"] == workers and row["run"] == repeat+1 for row in samples):
                continue
            details = measure(workers)
            if not details.pop("correct"):
                raise AssertionError(f"Experiment {experiment}, P={workers}: numerical result mismatch")
            row = {"P": workers, "run": repeat + 1, **details, "correct": True}
            samples.append(row)
            write_json(path, record)
            print(f"{experiment} | P={workers} | run={repeat+1} | T={row['time_ns']/1e9:.6f}s | verified", flush=True)
    medians = {p: statistics.median(row["time_ns"] for row in samples if row["P"] == p) for p in WORKERS}
    metrics = [{"P": p, "median_time_ns": medians[p], "time_s": medians[p] / 1e9,
                **scaling_metrics(medians[1], medians[p], p)} for p in WORKERS]
    record.update(status="COMPLETE", finished_at=timestamp(), metrics=metrics)
    write_json(path, record)
    for row in metrics:
        e = "n/a" if row["karp_flatt"] is None else f"{row['karp_flatt']:.6f}"
        print(f"P={row['P']} | median={row['time_s']:.6f}s | S={row['speedup']:.4f}x | E={row['efficiency_percent']:.2f}% | e={e}", flush=True)
    return record
