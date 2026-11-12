import argparse
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import emit_verify_tag, physical_cpu_count, run_benchmark

M = 24_000_000
STRIDES = (1, 2, 4, 8, 16, 32)


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        values = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        acc = values[slot_idx]
        for i in range(iters):
            acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        values[slot_idx] = acc
        del values
    finally:
        shm.close()


def measure(shm_name: str, processes: int, stride: int, iterations: int):
    def run() -> None:
        children = [
            mp.Process(target=fs_worker, args=(shm_name, worker * stride, iterations))
            for worker in range(processes)
        ]
        for child in children:
            child.start()
        for child in children:
            child.join()
            if child.exitcode != 0:
                raise RuntimeError(f"false-sharing worker failed: {child.exitcode}")

    return run_benchmark(run, warmup_runs=1, timed_runs=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a short validation workload")
    args = parser.parse_args()
    processes = physical_cpu_count()
    iterations = (M if not args.quick else 120_000) // processes
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        results = {}
        print(f"P_phys={processes}; P_log={mp.cpu_count()}; iterations/worker={iterations}")
        for stride in STRIDES:
            stats = measure(shm.name, processes, stride, iterations)
            throughput = processes * iterations / (stats.median_ms * 1000)
            results[stride] = stats
            print(f"stride={stride:2d} median_ms={stats.median_ms:.3f} "
                  f"iqr_ms={stats.iqr_ms:.3f} Mops/s={throughput:.3f}")

        scaling = {}
        for stride in (1, 16):
            baseline = None
            for count in range(1, (mp.cpu_count() or 1) + 1):
                stats = measure(shm.name, count, stride, max(1, M // count if not args.quick else 120_000 // count))
                baseline = stats.median_ms if baseline is None else baseline
                speedup = baseline / stats.median_ms
                efficiency = speedup / count
                scaling[(stride, count)] = stats.median_ms
                print(f"scale stride={stride} p={count} median_ms={stats.median_ms:.3f} "
                      f"speedup={speedup:.3f} efficiency={efficiency:.3f}")
        print(emit_verify_tag("PHASE1", "230103341",
                              scaling[(1, processes)], scaling[(16, processes)]))
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
