"""Strong/weak row decomposition of a five-point stencil, with real output arrays."""
import argparse
import csv
import ctypes
import math
import multiprocessing as mp
import statistics
import time
import traceback

import numpy as np

from common import (EVIDENCE, RESULTS, cpu_order, digest, native, params, pointer,
                    read_json, run_workers, write_json)
from export_schema import roofline_aliases


def worker(shared_input, shared_output, n, first, last, repeats, cpu, gate, queue):
    try:
        from common import pin_cpu
        pin_cpu(cpu)
        lib = native()
        source = np.ctypeslib.as_array(shared_input).reshape(n, n)
        output = np.ctypeslib.as_array(shared_output).reshape(n, n)
        lib.stencil(pointer(source), pointer(output), n, first, last)
        queue.put(("ready", cpu))
        if not gate.wait(timeout=180):
            raise TimeoutError("Start gate timeout")
        start = time.perf_counter_ns()
        for _ in range(repeats):
            lib.stencil(pointer(source), pointer(output), n, first, last)
        queue.put(("done", cpu, time.perf_counter_ns() - start))
    except BaseException:
        queue.put(("error", traceback.format_exc()))
        raise


def stencil_reference(source):
    result = np.zeros_like(source)
    result[1:-1, 1:-1] = (source[1:-1, 1:-1] + source[1:-1, :-2]
                           + source[1:-1, 2:] + source[:-2, 1:-1]
                           + source[2:, 1:-1]) * 0.2
    return result


def memory_bandwidth():
    # Copy payload exceeds shared L3; traffic = one read + one write (no RFO counted).
    source = np.ones(16_777_216, dtype=np.float64)
    target = np.zeros_like(source)
    for _ in range(5):
        np.copyto(target, source)
    times = []
    for _ in range(5):
        start = time.perf_counter_ns()
        np.copyto(target, source)
        times.append(time.perf_counter_ns() - start)
    if not np.array_equal(source, target):
        raise AssertionError("Copy calibration failed")
    return {"method": "Single-process NumPy copy, 128 MiB input + 128 MiB output, five warmups/five samples",
            "runs_ns": times, "median_ns": statistics.median(times),
            "traffic_bytes": source.nbytes * 2,
            "median_gb_s": source.nbytes * 2 / statistics.median(times),
            "kind": "empirical sustainable copy bandwidth, not theoretical hardware peak"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.runs < 1:
        parser.error("repeats and runs must be positive")
    settings = params()
    host = read_json(RESULTS / "host_profile.json")
    physical = host["cpu"]["physical_cores"]
    counts, p = [], 1
    while p <= physical:
        counts.append(p)
        p *= 2
    if counts[-1] != physical:
        counts.append(physical)
    cpus, rows, checks = cpu_order(), [], []
    context = mp.get_context("spawn")
    lib = native()
    for mode in ("strong", "weak"):
        for p in counts:
            n = settings["N"] if mode == "strong" else math.ceil(settings["N"] * math.sqrt(p))
            shared_input = context.RawArray(ctypes.c_double, n * n)
            shared_output = context.RawArray(ctypes.c_double, n * n)
            source = np.ctypeslib.as_array(shared_input).reshape(n, n)
            output = np.ctypeslib.as_array(shared_output).reshape(n, n)
            source[:] = np.random.default_rng(settings["seed"]).random((n, n))
            reference = stencil_reference(source)
            serial = np.zeros_like(source)
            lib.stencil(pointer(source), pointer(serial), n, 1, n-1)
            np.testing.assert_allclose(serial, reference, rtol=0, atol=1e-12)
            for repeat in range(1, args.runs + 1):
                output.fill(0)
                arguments = [(shared_input, shared_output, n,
                              1 + (n-2)*i//p, 1 + (n-2)*(i+1)//p,
                              args.repeats, cpus[i]) for i in range(p)]
                elapsed, lifecycle, completed = run_workers(worker, arguments)
                np.testing.assert_allclose(output, reference, rtol=0, atol=1e-12)
                output_digest = digest(output, settings["student_id"])
                rows.append({"mode": mode, "P": p, "N": n, "run": repeat,
                             "repeats": args.repeats, "work_cells": (n-2)**2 * args.repeats,
                             "time_ns": elapsed, "lifecycle_ns": lifecycle,
                             "max_worker_ns": max(message[2] for message in completed),
                             "checksum": float(np.sum(output)), "digest": output_digest,
                             "max_abs_error": float(np.max(np.abs(output-reference))),
                             "cpu_ids": ",".join(map(str, cpus[:p]))})
                checks.append({"mode": mode, "P": p, "N": n, "run": repeat,
                               "digest": output_digest, "reference_digest": digest(reference, settings["student_id"])})
                print(f"{mode:6} P={p} N={n} run={repeat}: {elapsed/1e6:.3f} ms", flush=True)
            # Keep one real array for the report; all run digests remain in telemetry.
            if mode == "strong" and p == counts[-1]:
                EVIDENCE.mkdir(exist_ok=True)
                np.save(EVIDENCE / "stencil_output.npy", output)
    medians = {(mode, p): statistics.median(r["time_ns"] for r in rows if r["mode"] == mode and r["P"] == p)
               for mode in ("strong", "weak") for p in counts}
    for row in rows:
        mode, p = row["mode"], row["P"]
        t1, tp = medians[mode, 1], medians[mode, p]
        ratio = t1 / tp
        w1 = (settings["N"]-2)**2 * args.repeats
        wp = row["work_cells"]
        row.update(median_time_ns=tp, speedup=ratio if mode == "strong" else "",
                   efficiency_percent=100*ratio/p if mode == "strong" else "",
                   effective_serial_fraction=(p/ratio-1)/(p-1) if mode == "strong" and p > 1 else "",
                   handout_scaled_speedup=p*w1*t1/(wp*tp) if mode == "weak" else "",
                   conventional_scaled_speedup=wp*t1/(w1*tp) if mode == "weak" else "",
                   weak_efficiency_percent=100*ratio if mode == "weak" else "",
                   cells_per_second=wp/(tp/1e9))
    with (RESULTS / "scaling_metrics.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    # Least squares in normalized time: TP/T1 - 1/P = s*(1 - 1/P).
    x = np.array([1-1/p for p in counts if p > 1])
    y = np.array([medians["strong", p]/medians["strong", 1]-1/p for p in counts if p > 1])
    serial_raw = float(np.dot(x, y) / np.dot(x, x)) if x.size else None
    serial_fit = min(1.0, max(0.0, serial_raw)) if serial_raw is not None else None
    fit = {"method": "Least-squares normalized wall time; raw and [0,1]-constrained estimates",
           "raw_serial_fraction": serial_raw, "fitted_serial_fraction": serial_fit,
           "asymptotic_speedup_ceiling": 1/serial_fit if serial_fit else None,
           "ceiling_note": "null if fitted s=0 or no multi-core samples; finite data do not establish infinite scaling",
           "time_ratio_rmse": float(np.sqrt(np.mean((y - serial_fit*x)**2))) if x.size else None}
    bandwidth = memory_bandwidth()
    peak_bandwidth = host.get("memory", {}).get("theoretical_peak_bandwidth_gb_s")
    # Explicit architectural model, not a measured instruction-throughput claim.
    peak_compute = (physical * host["cpu"]["base_frequency_mhz"] / 1000 * 16
                    if host["cpu"]["microarchitecture"] == "Zen 4" else None)
    best = max((r for r in rows if r["mode"] == "strong"), key=lambda r: r["cells_per_second"])
    measured_gflops = best["cells_per_second"] * 5 / 1e9
    write_json(EVIDENCE / "stencil_checksums.json", checks)
    write_json(RESULTS / "roofline_summary.json", roofline_aliases({
        "stencil": "Five-point average; 4 additions + 1 multiply per interior cell; zero output boundaries",
        "repeats_semantics": "100 independent evaluations of the same input, not 100 evolving timesteps; repeats is configurable",
        "flops_per_cell": 5, "naive_bytes_per_cell": 48,
        "naive_arithmetic_intensity_flop_per_byte": 5/48,
        "ideal_streaming_bytes_per_cell": 16,
        "ideal_streaming_arithmetic_intensity_flop_per_byte": 5/16,
        "write_allocate_streaming_bytes_per_cell": 24,
        "traffic_caveat": "48B is five loads plus one store before cache reuse; 16B assumes full input reuse and no write allocation; 24B includes read-for-ownership. None is a measured DRAM byte count.",
        "cache_summary": host["cache_summary"], "strong_input_output_bytes": 2*settings["N"]**2*8,
        "row_working_set_bytes_3_input_1_output": 4*settings["N"]*8,
        "bandwidth_calibration": bandwidth,
        "theoretical_peak_memory_bandwidth_gb_s": peak_bandwidth,
        "memory_peak_source": host.get("memory"),
        "theoretical_peak_compute_gflops": peak_compute,
        "compute_peak_model": "Zen 4 model: 2 x 256-bit FP64 FMA pipes x 4 lanes x 2 FLOPs = 16 FLOPs/core/cycle, times physical cores and registry nominal GHz. Assumes ideal vectorized FMA; scalar stencil is not expected to attain this.",
        "ridge_point_flop_per_byte": peak_compute/peak_bandwidth if peak_compute and peak_bandwidth else None,
        "theoretical_streaming_roof_gflops": peak_bandwidth*(5/16) if peak_bandwidth else None,
        "copy_based_streaming_roof_gflops": bandwidth["median_gb_s"]*(5/16),
        "best_measured_stencil_gflops": measured_gflops,
        "classification": "Memory-Bandwidth Bound in the theoretical streaming Roofline model" if peak_compute and peak_bandwidth and 5/16 < peak_compute/peak_bandwidth else "Low-arithmetic-intensity, memory/cache-bandwidth-sensitive kernel",
        "classification_scope": "AI is below the model ridge point; this is not proof of saturated DRAM in the scalar implementation. Cache reuse, scalar instruction throughput, barriers and ctypes overhead also affect runtime. DRAM traffic is modelled, not counted.",
        "amdahl_fit": fit,
        "gustafson_scaled_speedup_at_max_P": physical - serial_fit*(physical-1) if serial_fit is not None else None,
        "gustafson_caveat": "Model projection using fitted strong serial fraction, not an independently measured weak serial fraction",
        "weak_scaling_formula_note": "Handout P*W1*T1/(WP*TP) tends to T1/TP for WP=P*W1 (weak efficiency). Conventional scaled speedup is WP*T1/(W1*TP); both are logged.",
        "timing_scope": "Start-event release through completion messages; full spawn/setup/join lifecycle separately recorded",
        "process_counts": counts}))
    print("Amdahl fit:", fit)
    print("Copy bandwidth GB/s:", bandwidth["median_gb_s"])


if __name__ == "__main__":
    main()
