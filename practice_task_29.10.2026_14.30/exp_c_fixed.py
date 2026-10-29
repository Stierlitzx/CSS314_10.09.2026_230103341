"""Experiment C: private accumulation followed by one final reduction."""
import multiprocessing as mp
import os
import time

from benchmark_support import parser, partition, positive, run_suite


def private_worker(iterations):
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def measure_privatized(p, total_increments=20_000_000):
    tasks = partition(total_increments, p)
    context = mp.get_context("spawn")
    start = time.perf_counter_ns()
    with context.Pool(processes=p) as pool:
        partials = pool.map(private_worker, tasks)
        total = sum(partials)
    elapsed = time.perf_counter_ns() - start
    return {"time_ns": elapsed, "assigned_increments": tasks, "partial_totals": partials,
            "actual_total": total, "expected_total": total_increments,
            "correct": partials == tasks and total == total_increments}


def run_privatized(p, total_increments=20_000_000):
    return measure_privatized(p, total_increments)["time_ns"] / 1e9


def worker_ready(queue):
    queue.put(os.getpid())


def measure_prestarted(p, total_increments=20_000_000):
    """Supplementary timing only: exclude pool startup and shutdown explicitly."""
    tasks = partition(total_increments, p)
    context = mp.get_context("spawn")
    ready = context.Queue()
    lifecycle_start = time.perf_counter_ns()
    try:
        with context.Pool(processes=p, initializer=worker_ready, initargs=(ready,)) as pool:
            worker_pids = [ready.get(timeout=30) for _ in range(p)]
            start = time.perf_counter_ns()
            partials = pool.map(private_worker, tasks)
            total = sum(partials)
            elapsed = time.perf_counter_ns()-start
        return {"time_ns": elapsed, "lifecycle_ns": time.perf_counter_ns()-lifecycle_start,
                "worker_pids": worker_pids, "assigned_increments": tasks, "partial_totals": partials,
                "actual_total": total, "expected_total": total_increments,
                "correct": len(set(worker_pids)) == p and partials == tasks and total == total_increments}
    finally:
        ready.close()
        ready.join_thread()


def main():
    cli = parser(__doc__)
    cli.add_argument("--total-increments", type=positive, default=20_000_000)
    cli.add_argument("--steady-state", action="store_true", help="Supplementary prestarted-pool timing; does not replace default end-to-end results")
    args = cli.parse_args()
    workload = {"total_increments": args.total_increments,
                "synchronization": "Zero shared writes in the loop; one final parent sum over worker results"}
    if args.steady_state:
        run_suite("C-steady", lambda p: measure_prestarted(p, args.total_increments), workload, args.runs,
                  timing_scope="Supplementary: prestarted ready pool; includes map, IPC, private loops and reduction; excludes pool startup/shutdown")
    else:
        run_suite("C", lambda p: measure_privatized(p, args.total_increments), workload, args.runs)


if __name__ == "__main__":
    main()
