"""Query actual Windows CPU/cache topology, not guessed cache sizes."""
import argparse
import ctypes
import json
import os
import platform
import sys
import subprocess
import time

from common import RESULTS, timestamp, write_json
from export_schema import host_aliases
from salt import get_student_params


def windows_topology():
    import winreg

    class Cache(ctypes.Structure):
        _fields_ = [("level", ctypes.c_byte), ("associativity", ctypes.c_byte),
                    ("line_size", ctypes.c_ushort), ("size", ctypes.c_ulong),
                    ("kind", ctypes.c_int)]

    class Details(ctypes.Union):
        _fields_ = [("flags", ctypes.c_byte), ("cache", Cache),
                    ("reserved", ctypes.c_ulonglong * 2)]

    class Info(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int),
                    ("details", Details)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetLogicalProcessorInformation
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    size = ctypes.c_ulong()
    query(None, ctypes.byref(size))
    if not size.value:
        raise ctypes.WinError(ctypes.get_last_error())
    storage = ctypes.create_string_buffer(size.value)
    if not query(storage, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    records = ctypes.cast(storage, ctypes.POINTER(Info))
    cores, caches = [], []
    for index in range(size.value // ctypes.sizeof(Info)):
        record = records[index]
        cpus = [i for i in range(64) if record.mask & (1 << i)]
        if record.relationship == 0:
            cores.append(cpus)
        elif record.relationship == 2:
            cache = record.details.cache
            caches.append({"level": cache.level,
                           "type": {0: "unified", 1: "instruction", 2: "data"}.get(cache.kind, "other"),
                           "size_bytes": cache.size, "line_size_bytes": cache.line_size,
                           "logical_cpu_ids": cpus})
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        identifier = winreg.QueryValueEx(key, "Identifier")[0]
        mhz = winreg.QueryValueEx(key, "~MHz")[0]
    # This mapping is explicitly derived from the model, not an OS measurement.
    microarch = "Zen 4" if "Ryzen 5 7640HS" in model else None
    return {"model": model, "identifier": identifier, "microarchitecture": microarch,
            "microarchitecture_source": "AMD model mapping" if microarch else "unavailable",
            "base_frequency_mhz": mhz, "frequency_source": "Windows registry ~MHz (reported nominal)",
            "physical_cores": len(cores), "logical_threads": sum(map(len, cores)),
            "smt_present": any(len(c) > 1 for c in cores), "core_logical_cpu_ids": cores}, caches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", default="230103341")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This host profiler targets the measured Windows lab host; port topology queries before running elsewhere")
    cpu, caches = windows_topology()
    query = ("Get-CimInstance Win32_PhysicalMemory | Select-Object BankLabel, DeviceLocator, "
             "ConfiguredClockSpeed, Speed, DataWidth, TotalWidth, Capacity, SMBIOSMemoryType | ConvertTo-Json -Depth 3")
    memory = {"source": "Win32_PhysicalMemory via Get-CimInstance", "modules": [],
              "theoretical_peak_bandwidth_gb_s": None}
    try:
        modules = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True))
        modules = modules if isinstance(modules, list) else [modules]
        memory["modules"] = modules
        banks = {module["BankLabel"] for module in modules}
        if (len(banks) == len(modules) and all("CHANNEL" in m["BankLabel"] and
                m["SMBIOSMemoryType"] == 34 and m["DataWidth"] == 64 and m["ConfiguredClockSpeed"] > 0 for m in modules)):
            memory["theoretical_peak_bandwidth_gb_s"] = sum(m["ConfiguredClockSpeed"]*m["DataWidth"]/8/1000 for m in modules)
        memory["bandwidth_model"] = ("Sum distinct firmware-labelled DDR5 channels: configured rate (interpreted as MT/s) * data width / 8. "
                                      "Payload upper bound, not measured sustained bandwidth; assumes labelled channels are independent.")
    except (subprocess.SubprocessError, ValueError, KeyError) as error:
        memory["query_error"] = str(error)
    summary = {}
    for name, level, kinds in (("l1_data_bytes_per_cache", 1, ("data",)),
                                ("l2_bytes_per_cache", 2, ("unified", "data")),
                                ("l3_bytes_per_cache", 3, ("unified",))):
        summary[name] = sorted({c["size_bytes"] for c in caches if c["level"] == level and c["type"] in kinds})
    summary["cache_line_sizes_bytes"] = sorted({c["line_size_bytes"] for c in caches})
    write_json(RESULTS / "host_profile.json", host_aliases({
        "captured_at": timestamp(), "topology_source": "Windows GetLogicalProcessorInformation",
        "cpu": cpu, "cache_summary": summary, "caches": caches, "memory": memory,
        "os": {"system": platform.system(), "release": platform.release(), "kernel_version": platform.version()},
        "python": {"implementation": platform.python_implementation(), "version": sys.version,
                   "executable": sys.executable, "architecture": platform.machine()},
        "timer": {"name": "perf_counter_ns", "resolution_seconds": time.get_clock_info("perf_counter").resolution},
        "hardware_performance_counters": {"cache_hits": None, "cache_misses": None,
                                          "status": "Not collected; timing measurements do not measure cache hit rates"}}))
    salted = get_student_params(args.student_id)
    write_json(RESULTS / "salted_params.json", salted)
    print(cpu)
    print(summary)
    print("Memory:", memory)
    print(salted)


if __name__ == "__main__":
    main()
