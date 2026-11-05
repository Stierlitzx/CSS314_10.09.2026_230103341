"""Task 22: real spawn-process weak scaling with matched serial references."""
import argparse
import csv
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["run_id", "timestamp", "P", "run", "k", "total_points", "serial_seconds",
          "parallel_seconds", "scaled_speedup", "serial_integral", "parallel_integral",
          "absolute_pi_error", "worker_pids", "measurement_order", "correct"]


def timestamp():
    return datetime.now().astimezone().isoformat()


def integrate(start, stop, total_points):
    """Composite midpoint quadrature on one nonoverlapping interval of indices."""
    subtotal = 0.0
    for i in range(start, stop):
        x = (i + 0.5) / total_points
        subtotal += 4.0 / (1.0 + x * x)
    return subtotal / total_points


def child(send, start, stop, total_points):
    try:
        send.send((os.getpid(), integrate(start, stop, total_points)))
    finally:
        send.close()


def serial_reference(p, k):
    start = time.perf_counter_ns()
    value = math.fsum(integrate(rank * k, (rank + 1) * k, p * k) for rank in range(p))
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return elapsed, value


def parallel_measure(p, k):
    """Time allocation, spawn, work, pipes, reduction, join and handle cleanup."""
    context = mp.get_context("spawn")
    processes, receivers = [], []
    start = time.perf_counter_ns()
    try:
        for rank in range(p):
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=child, args=(send, rank * k, (rank + 1) * k, p * k))
            receivers.append(receive)
            processes.append(process)
            try:
                process.start()
            finally:
                send.close()
        results = []
        for receive in receivers:
            if not receive.poll(120):
                raise TimeoutError("Worker exceeded 120 seconds; no timing accepted")
            results.append(receive.recv())
        value = math.fsum(result[1] for result in results)
        for process in processes:
            process.join(timeout=120)
            if process.exitcode != 0:
                raise RuntimeError(f"Worker failed: exitcode={process.exitcode}")
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
                process.close()
        for receive in receivers:
            receive.close()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return elapsed, value, [result[0] for result in results]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def core_counts(value):
    try:
        counts = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers") from error
    if not counts or not all(1 <= p <= (os.cpu_count() or 1) for p in counts):
        raise argparse.ArgumentTypeError("counts must be between 1 and host hardware concurrency")
    return counts


def capture_host():
    host = dict(captured_at=timestamp(), student_id="230103341", os=platform.platform(),
                python=sys.version, python_executable=sys.executable,
                logical_processors=os.cpu_count(), cpu=platform.processor(),
                start_method="spawn", affinity="OS managed; unchanged", timer="perf_counter_ns",
                timer_resolution=time.get_clock_info("perf_counter").resolution,
                temperature="not instrumented", numa="not instrumented",
                memory_channels_populated="not independently verified")
    if os.name == "nt":
        probes = {
            "processor_cim": "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize,MaxClockSpeed | ConvertTo-Json",
            "system_cim": "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | ConvertTo-Json",
            "memory_cim": "Get-CimInstance Win32_PhysicalMemory | Select-Object DeviceLocator,Capacity,Speed,ConfiguredClockSpeed | ConvertTo-Json",
            "os_cim": "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | ConvertTo-Json",
        }
        for name, command in probes.items():
            try:
                host[name] = json.loads(subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", command], text=True, timeout=30))
            except (subprocess.SubprocessError, ValueError, OSError) as error:
                host[name] = {"error": str(error)}
    return host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=core_counts, default=list(range(1, (os.cpu_count() or 1) + 1)))
    parser.add_argument("--k", type=positive, default=1000000, help="fixed midpoint evaluations per process")
    parser.add_argument("--runs", type=positive, default=5)
    parser.add_argument("--seed", type=int, default=230103341, help="configuration-order randomization only")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results.csv")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing dataset")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; choose another --output or use --overwrite")
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata = dict(run_id=str(uuid.uuid4()), started_at=timestamp(), status="running",
                    workers=args.workers, runs=args.runs, k=args.k, seed=args.seed,
                    host=capture_host(),
                    timing_scope="End-to-end process creation, spawn, work, IPC, reduction, join, cleanup; correctness checks excluded",
                    reference="Direct sequential execution of identical P chunks, without process startup")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    integrate(0, 10000, 10000)  # Parent interpreter warmup; not a retained sample.
    rng = random.Random(args.seed)
    try:
        with args.output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for run in range(1, args.runs + 1):
                order = args.workers.copy()
                rng.shuffle(order)
                for p in order:
                    captured_at = timestamp()
                    if run % 2:
                        ts, serial = serial_reference(p, args.k)
                        tp, parallel, pids = parallel_measure(p, args.k)
                    else:
                        tp, parallel, pids = parallel_measure(p, args.k)
                        ts, serial = serial_reference(p, args.k)
                    tolerance = max(1e-10, 2 / (p * args.k) ** 2)
                    correct = math.isclose(serial, parallel, rel_tol=1e-12, abs_tol=1e-12)
                    correct = correct and abs(parallel - math.pi) <= tolerance and len(set(pids)) == p
                    if not correct:
                        raise AssertionError("Matched-work checksum, quadrature or worker count failed")
                    row = dict(run_id=metadata["run_id"], timestamp=captured_at, P=p, run=run,
                               k=args.k, total_points=p * args.k, serial_seconds=ts,
                               parallel_seconds=tp, scaled_speedup=ts / tp, serial_integral=serial,
                               parallel_integral=parallel, absolute_pi_error=abs(parallel - math.pi),
                               worker_pids=";".join(map(str, pids)),
                               measurement_order="serial-first" if run % 2 else "parallel-first", correct=correct)
                    writer.writerow(row)
                    handle.flush()
                    print(f"P={p:2} run={run} Ts={ts:.6f}s Tp={tp:.6f}s S={ts/tp:.3f} correct={correct}", flush=True)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = repr(error)
        raise
    finally:
        metadata["finished_at"] = timestamp()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()