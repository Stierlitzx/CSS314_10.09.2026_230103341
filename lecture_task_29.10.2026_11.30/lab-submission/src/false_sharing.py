"""Shared-memory contention, cache-line padding, and private accumulation."""
import argparse
import csv
import ctypes
import multiprocessing as mp
import statistics
import time
import traceback

from common import EVIDENCE, RESULTS, cpu_order, native, read_json, run_workers, write_json
from export_schema import sharing_aliases


def worker(shared, offset, index, iterations, mode, backend, cpu, lock, gate, queue):
    try:
        from common import pin_cpu
        pin_cpu(cpu)
        lib = native() if backend == "native" else None
        counters = ctypes.cast(ctypes.byref(shared, offset * 8), ctypes.POINTER(ctypes.c_int64))
        private = (ctypes.c_int64 * 1)()
        queue.put(("ready", cpu))
        if not gate.wait(timeout=180):
            raise TimeoutError("Start gate timeout")
        start = time.perf_counter_ns()
        if backend == "native":
            if mode == "privatized":
                lib.counter_loop(private, 0, iterations)
                with lock:
                    counters[index] = private[0]
            else:
                lib.counter_loop(counters, index, iterations)
        elif mode == "privatized":
            value = 0
            for _ in range(iterations):
                value += 1
            with lock:
                counters[index] = value
        else:
            for _ in range(iterations):
                counters[index] += 1
        queue.put(("done", cpu, time.perf_counter_ns() - start))
    except BaseException:
        queue.put(("error", traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5_000_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    args = parser.parse_args()
    if args.iterations < 1 or args.runs < 1:
        parser.error("iterations and runs must be positive")
    cpus = cpu_order()
    if len(cpus) < 8:
        raise RuntimeError("The required P=8 experiment needs at least eight logical CPUs")
    line_sizes = read_json(RESULTS / "host_profile.json")["cache_summary"]["cache_line_sizes_bytes"]
    if line_sizes != [64]:
        raise RuntimeError("The prescribed padding experiment requires measured 64-byte cache lines")
    context = mp.get_context("spawn")
    rows, counter_runs = [], []
    for processes in (1, 2, 4, 8):
        for repeat in range(1, args.runs + 1):
            modes = ["unpadded", "padded", "privatized"]
            modes = modes[repeat % 3:] + modes[:repeat % 3]
            for mode in modes:
                # Extra slots allow explicit 64-byte alignment, regardless of allocator alignment.
                shared = context.RawArray(ctypes.c_int64, 8 * processes + 8)
                offset = ((-ctypes.addressof(shared)) % 64) // 8
                spacing = 8 if mode == "padded" else 1
                indices = [p * spacing for p in range(processes)]
                lock = context.Lock()  # Only used for the one final privatized write.
                arguments = [(shared, offset, indices[p], args.iterations, mode, args.backend,
                              cpus[p], lock) for p in range(processes)]
                elapsed, lifecycle, completed = run_workers(worker, arguments)
                values = [int(shared[offset + i]) for i in indices]
                if values != [args.iterations] * processes:
                    raise AssertionError(f"Incorrect counters: {values}")
                rows.append({"mode": mode, "P": processes, "run": repeat,
                             "iterations_per_worker": args.iterations, "backend": args.backend,
                             "time_ns": elapsed, "lifecycle_ns": lifecycle,
                             "max_worker_ns": max(message[2] for message in completed),
                             "total_count": sum(values), "cpu_ids": ",".join(map(str, cpus[:processes])),
                             "base_address_mod_64": (ctypes.addressof(shared) + offset * 8) % 64})
                counter_runs.append({"mode": mode, "P": processes, "run": repeat,
                                     "iterations_per_worker": args.iterations, "counts": values,
                                     "indices": indices, "counter_offsets_bytes": [i * 8 for i in indices]})
                print(f"{mode:10} P={processes} run={repeat}: {elapsed/1e6:.3f} ms; total={sum(values)}", flush=True)
    medians = {(mode, p): statistics.median(r["time_ns"] for r in rows if r["mode"] == mode and r["P"] == p)
               for mode in ("unpadded", "padded", "privatized") for p in (1, 2, 4, 8)}
    for row in rows:
        p, mode = row["P"], row["mode"]
        t1, tp = medians[mode, 1], medians[mode, p]
        speedup = t1 / tp
        row.update(median_time_ns=tp, handout_speedup=speedup,
                   karp_flatt_handout=(p / speedup - 1) / (p - 1) if p > 1 else "",
                   throughput_speedup=p * speedup,
                   work_normalized_e=(1 / speedup - 1) / (p - 1) if p > 1 else "")
    rows = sharing_aliases(rows)
    with (RESULTS / "false_sharing_karp_flatt.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_json(EVIDENCE / "counter_outputs.json", counter_runs)
    write_json(EVIDENCE / "false_sharing_method.json", {
        "backend": args.backend, "iterations_per_worker": args.iterations, "runs": args.runs,
        "allocation": "64-byte-aligned multiprocessing.RawArray(c_int64); no lock on per-iteration accesses",
        "native_semantics": "One volatile int64 load/add/store per iteration; privatized counter uses process-private memory",
        "cpu_selection": "One logical CPU per physical core first, then SMT siblings; P=8 exceeds six physical cores on this host",
        "timing": "Start gate through all completion messages; spawn/import/setup/join excluded. lifecycle_ns includes them.",
        "caveat": "K is per worker, so total work grows with P. Handout e uses T1/TP and is not a strong-scaling serial fraction. Even ideal flat TP yields e=1. work_normalized_e uses throughput speedup P*T1/TP."})


if __name__ == "__main__":
    main()
