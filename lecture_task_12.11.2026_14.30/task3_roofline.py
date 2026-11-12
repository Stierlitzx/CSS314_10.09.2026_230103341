import argparse
import multiprocessing as mp
import time

from harness import emit_verify_tag


def copy_worker(size):
    source = bytearray(size)
    target = bytearray(size)
    target[:] = source


def fma_worker(iterations):
    x = 1.001
    for _ in range(iterations):
        x = x * 1.0000001 + 0.0000001
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    workers = min(physical_count(), 4) if args.quick else physical_count()
    size = 1 << 20 if args.quick else 96 * 10**6
    start = time.perf_counter_ns()
    with mp.Pool(workers) as pool:
        pool.map(copy_worker, [size] * workers)
    copy_seconds = (time.perf_counter_ns() - start) / 1e9
    bandwidth = 2 * size * workers / copy_seconds / 1e9
    iterations = 10_000 if args.quick else 80_000_000
    start = time.perf_counter_ns()
    with mp.Pool(workers) as pool:
        pool.map(fma_worker, [iterations] * workers)
    compute_seconds = (time.perf_counter_ns() - start) / 1e9
    gflops = 2 * iterations * workers / compute_seconds / 1e9
    print(f"workers={workers} bandwidth_GBps={bandwidth:.4f} compute_GFLOPS={gflops:.4f}")
    print(f"ridge_point_FLOP_per_byte={gflops / bandwidth:.6f}")
    print(emit_verify_tag("PHASE3", "230103341", bandwidth, gflops))


def physical_count():
    return max(1, (mp.cpu_count() or 1) // 2)


if __name__ == "__main__":
    mp.freeze_support()
    main()
