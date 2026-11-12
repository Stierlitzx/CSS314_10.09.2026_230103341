import multiprocessing as mp
import json
import platform
import subprocess
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
print(f"Python/OS: {platform.python_version()} / {platform.platform()}")
if platform.system() == "Windows":
    query = ("Get-CimInstance Win32_Processor | "
             "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
             "L2CacheSize,L3CacheSize | ConvertTo-Json -Compress")
    completed = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                               capture_output=True, text=True, check=True)
    topology = json.loads(completed.stdout)
    print(f"CPU model: {topology.get('Name', 'unknown')}")
    print(f"L2 cache KB: {topology.get('L2CacheSize', 'unknown')}")
    print(f"L3 cache KB: {topology.get('L3CacheSize', 'unknown')}")
    print("Cache line bytes: 64 (x86-64 default; verify with native OS tools)")

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
