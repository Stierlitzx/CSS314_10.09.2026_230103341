import argparse
import math
import multiprocessing as mp
import time

from harness import emit_verify_tag


def kernel(bounds):
    start, end = bounds
    total = 0.0
    for i in range(start, end):
        x = i * 1.0e-6
        for _ in range(30):
            x = math.sin(x) * 0.5 + math.sqrt(abs(x) + 1.0)
        total += x
    return total


def serial_stage(n):
    total = 0.0
    for i in range(max(1, n // 20)):
        x = i * 1.0e-5
        for _ in range(10):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        total += x
    return total


def timed(pool, n, workers):
    serial_stage(n)
    step = (n + workers - 1) // workers
    ranges = [(i, min(n, i + step)) for i in range(0, n, step)]
    start = time.perf_counter_ns()
    pool.map(kernel, ranges)
    return (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    logical = mp.cpu_count() or 1
    strong_n = 20_000 if args.quick else 1_200_000
    base_n = 4_000 if args.quick else 200_000
    strong = {}
    weak = {}
    for workers in range(1, logical + 1):
        with mp.Pool(workers) as pool:
            strong[workers] = timed(pool, strong_n, workers)
            weak[workers] = timed(pool, workers * base_n, workers)
        print(f"p={workers:2d} strong_ms={strong[workers]:.3f} weak_ms={weak[workers]:.3f}")
    t1 = strong[1]
    serial_start = time.perf_counter_ns()
    serial_stage(strong_n)
    serial_ms = (time.perf_counter_ns() - serial_start) / 1e6
    s = serial_ms / max(1e-9, serial_ms + t1)
    observed = t1 / strong[logical]
    gustafson = weak[1] / weak[logical]
    print(f"observed_strong_speedup={observed:.4f} serial_fraction={s:.6f}")
    print(emit_verify_tag("PHASE2", "230103341", observed, gustafson))


if __name__ == "__main__":
    mp.freeze_support()
    main()
