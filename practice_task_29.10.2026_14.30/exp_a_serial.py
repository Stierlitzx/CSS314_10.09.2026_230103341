"""Experiment A: fixed total compute work preceded by a 0.40-second serial pause."""
import math
import multiprocessing as mp
import time

from benchmark_support import parser, positive, run_suite


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def measure_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    if workers < 1 or total_chunks < 1 or chunk_size < 1:
        raise ValueError("workers, total_chunks and chunk_size must be positive")
    context = mp.get_context("spawn")
    start = time.perf_counter_ns()
    time.sleep(0.40)
    pause_end = time.perf_counter_ns()
    with context.Pool(processes=workers) as pool:
        values = pool.map(heavy_math, [chunk_size] * total_chunks)
    elapsed = time.perf_counter_ns() - start
    # Independent recomputation and validation are outside the timed interval.
    expected_chunk = heavy_math(chunk_size)
    correct = len(values) == total_chunks and all(math.isclose(value, expected_chunk, rel_tol=1e-12, abs_tol=1e-12) for value in values)
    return {"time_ns": elapsed, "actual_pause_ns": pause_end-start,
            "chunk_values": values, "checksum": math.fsum(values),
            "reference_checksum": math.fsum([expected_chunk] * total_chunks), "correct": correct}


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    return measure_experiment(workers, total_chunks, chunk_size)["time_ns"] / 1e9


def main():
    cli = parser(__doc__)
    cli.add_argument("--total-chunks", type=positive, default=16)
    cli.add_argument("--chunk-size", type=positive, default=2_000_000)
    args = cli.parse_args()
    run_suite("A", lambda p: measure_experiment(p, args.total_chunks, args.chunk_size),
              {"total_chunks": args.total_chunks, "chunk_size": args.chunk_size,
               "math_iterations": args.total_chunks*(args.chunk_size-1), "serial_pause_s": 0.40}, args.runs)


if __name__ == "__main__":
    main()
