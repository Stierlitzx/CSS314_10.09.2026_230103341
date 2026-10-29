"""Instructor-schema aliases derived from the original recorded measurements.

Run once to upgrade existing exports without rerunning or changing raw samples.
The benchmark generators use the same functions for all subsequent runs.
"""
import csv
import statistics

from common import RESULTS, read_json, write_json


def host_aliases(profile):
    lines = profile["cache_summary"]["cache_line_sizes_bytes"]
    if len(lines) != 1:
        raise ValueError("Cannot export a single cache-line size from heterogeneous telemetry")
    return dict(profile, cpu_model=profile["cpu"]["model"],
                physical_cores=profile["cpu"]["physical_cores"],
                logical_threads=profile["cpu"]["logical_threads"], cache_line_bytes=lines[0])


def dense_aliases(result):
    return dict(result, naive_time_s=result["median_ns"]["ijk"] / 1e9,
                interchanged_time_s=result["median_ns"]["ikj"] / 1e9)


def sharing_aliases(rows):
    for row in rows:
        p = int(row["P"])
        for mode in ("unpadded", "padded"):
            baseline = statistics.median(int(r["time_ns"]) for r in rows if r["mode"] == mode and int(r["P"]) == 1)
            elapsed = statistics.median(int(r["time_ns"]) for r in rows if r["mode"] == mode and int(r["P"]) == p)
            row["e_" + mode] = (p*elapsed/baseline-1)/(p-1) if p > 1 else ""
    return rows


def roofline_aliases(result):
    return dict(result, arithmetic_intensity=result["naive_arithmetic_intensity_flop_per_byte"],
                regime="memory-bound" if "Memory-Bandwidth Bound" in result["classification"] else "compute-bound")


def main():
    write_json(RESULTS / "host_profile.json", host_aliases(read_json(RESULTS / "host_profile.json")))
    write_json(RESULTS / "loop_interchange.json", dense_aliases(read_json(RESULTS / "loop_interchange.json")))
    write_json(RESULTS / "roofline_summary.json", roofline_aliases(read_json(RESULTS / "roofline_summary.json")))
    path = RESULTS / "stride_results.tsv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns, rows = list(reader.fieldnames), list(reader)
    if "latency_ns" not in columns:
        columns.append("latency_ns")
    for row in rows:
        row["latency_ns"] = row["ns_per_element"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    path = RESULTS / "false_sharing_karp_flatt.tsv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = sharing_aliases(list(csv.DictReader(stream, delimiter="\t")))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("Instructor M0-M3 schema aliases exported; original hardware capture time and raw timing samples retained.")


if __name__ == "__main__":
    main()
