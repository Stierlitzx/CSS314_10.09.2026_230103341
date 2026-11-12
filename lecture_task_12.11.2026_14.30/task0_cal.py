import multiprocessing as mp
import time
from harness import emit_verify_tag, physical_cpu_count

# Phase 0: Hardware Topology Audit & Harness Calibration
print("=" * 60)
print("PHASE 0: Hardware Topology Audit")
print("=" * 60)

# Query CPU count
cpu_count = mp.cpu_count()
print(f"Logical CPU cores (P_log): {cpu_count}")
print(f"Physical CPU cores (P_phys): {physical_cpu_count()}")

# Measure average call latency over 2M clock reads
N = 2_000_000
calls = []
for _ in range(5):
    t0 = time.perf_counter_ns()
    for i in range(N):
        _ = time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    calls.append((t1 - t0) / N)

avg_latency_ns = sum(calls) / len(calls)
print(f"Average call latency: {avg_latency_ns:.4f} ns")

# We'll need to compute the verify hash for phase 0
# The spec says to pass student_seed. We'll use a fixed seed "230103341"
student_seed = "230103341"
verify = emit_verify_tag("PHASE0", student_seed, avg_latency_ns, cpu_count)
print(verify)

print(f"\nPhase 0 Complete!")
