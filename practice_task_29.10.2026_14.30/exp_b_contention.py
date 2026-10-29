"""Experiment B: every increment takes the same shared mutex."""
import multiprocessing as mp
import time

from benchmark_support import parser, partition, positive, run_suite


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:
            shared_counter.value += 1


def measure_contended(p, total_increments=2_000_000):
    tasks = partition(total_increments, p)
    if total_increments > 2_147_483_647:
        raise ValueError("The handout's signed 32-bit counter cannot hold this total")
    context = mp.get_context("spawn")
    # Preserve the handout's synchronized Value plus explicit shared mutex.
    counter, lock = context.Value("i", 0), context.Lock()
    processes = [context.Process(target=contended_worker, args=(counter, lock, count)) for count in tasks]
    try:
        start = time.perf_counter_ns()
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
            if process.exitcode != 0:
                raise RuntimeError(f"Counter worker failed or timed out: {process.exitcode}")
        elapsed = time.perf_counter_ns() - start
        actual = counter.value
        return {"time_ns": elapsed, "assigned_increments": tasks, "actual_total": actual,
                "expected_total": total_increments, "correct": actual == total_increments}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def run_contended(p, total_increments=2_000_000):
    return measure_contended(p, total_increments)["time_ns"] / 1e9


def main():
    cli = parser(__doc__)
    cli.add_argument("--total-increments", type=positive, default=2_000_000)
    args = cli.parse_args()
    run_suite("B", lambda p: measure_contended(p, args.total_increments),
              {"total_increments": args.total_increments, "counter": "multiprocessing.Value('i', 0)",
               "synchronization": "One explicit shared Lock per increment; Value retains its default internal lock"}, args.runs)


if __name__ == "__main__":
    main()
