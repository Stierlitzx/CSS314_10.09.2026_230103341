"""True i-j-k and i-k-j loops on an M=N/2 sub-block with row pitch N."""
import statistics
import time

import numpy as np

from common import EVIDENCE, RESULTS, cpu_order, digest, native, params, pin_cpu, pointer, write_json
from export_schema import dense_aliases


def main():
    settings = params()
    n, m = settings["N"], settings["N"] // 2
    rng = np.random.default_rng(settings["seed"])
    a, b = rng.random((n, n)), rng.random((n, n))
    lib = native()
    pin_cpu(cpu_order()[0])
    outputs, timings = {}, {"ijk": [], "ikj": []}
    # Warm up code and input pages outside the timed samples.
    for name in timings:
        c = np.zeros((m, m), dtype=np.float64)
        getattr(lib, "matmul_" + name)(pointer(a), pointer(b), pointer(c), n, m)
    for repeat in range(3):
        # Alternate order to reduce systematic first/second thermal bias.
        for name in (("ijk", "ikj") if repeat % 2 == 0 else ("ikj", "ijk")):
            c = np.zeros((m, m), dtype=np.float64)
            start = time.perf_counter_ns()
            getattr(lib, "matmul_" + name)(pointer(a), pointer(b), pointer(c), n, m)
            timings[name].append(time.perf_counter_ns() - start)
            outputs[name] = c
    np.testing.assert_allclose(outputs["ijk"], outputs["ikj"], rtol=0, atol=1e-12)
    reference = a[:m, :m] @ b[:m, :m]
    np.testing.assert_allclose(outputs["ijk"], reference, rtol=0, atol=1e-12)
    EVIDENCE.mkdir(exist_ok=True)
    np.savez_compressed(EVIDENCE / "dense_outputs.npz", ijk=outputs["ijk"], ikj=outputs["ikj"], reference=reference)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    result = {"N": n, "M": m, "row_pitch_elements": n, "seed": settings["seed"],
              "backend": "ctypes C99 scalar loops", "runs_ns": timings, "median_ns": medians,
              "speedup": medians["ijk"] / medians["ikj"],
              "max_abs_difference": float(np.max(np.abs(outputs["ijk"] - outputs["ikj"]))),
              "max_abs_error_vs_numpy": float(np.max(np.abs(outputs["ijk"] - reference))),
              "digest": {name: digest(value, settings["student_id"]) for name, value in outputs.items()}}
    result = dense_aliases(result)
    write_json(RESULTS / "loop_interchange.json", result)
    print(result)


if __name__ == "__main__":
    main()
