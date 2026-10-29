"""Scalar native reduction: five warmups and five raw samples per stride."""
import csv
import statistics
import time

import numpy as np

from common import RESULTS, cpu_order, native, params, pin_cpu, pointer


def main():
    data = np.ones(params()["buffer_elements"], dtype=np.float64)
    lib = native()
    pin_cpu(cpu_order()[0])
    with (RESULTS / "stride_results.tsv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["stride", "elements_touched", "median_time_ns", "ns_per_element", "useful_gb_s", "sum"] + [f"run_{i}_ns" for i in range(1, 6)] + ["latency_ns"]
        writer = csv.DictWriter(stream, columns, delimiter="\t")
        writer.writeheader()
        for stride in (1, 2, 4, 8, 16, 32, 64, 128):
            touched = len(range(0, data.size, stride))
            for _ in range(5):
                lib.stride_sum(pointer(data), data.size, stride)
            runs = []
            for _ in range(5):
                start = time.perf_counter_ns()
                total = lib.stride_sum(pointer(data), data.size, stride)
                runs.append(time.perf_counter_ns() - start)
                if total != touched:
                    raise AssertionError("Stride reduction mismatch")
            elapsed = statistics.median(runs)
            row = dict(stride=stride, elements_touched=touched, median_time_ns=elapsed,
                       ns_per_element=elapsed/touched, latency_ns=elapsed/touched,
                       useful_gb_s=touched*8/elapsed, sum=total)
            row.update({f"run_{i}_ns": value for i, value in enumerate(runs, 1)})
            writer.writerow(row)
            print(f"stride {stride:3}: {elapsed/touched:.3f} ns/element")


if __name__ == "__main__":
    main()
