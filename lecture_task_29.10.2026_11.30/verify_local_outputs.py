"""Supplementary numerical checks; instructor receipt is generated separately."""
import csv
import hashlib
import math
from pathlib import Path
import statistics
import sys

import numpy as np

LECTURE = Path(__file__).resolve().parent
ROOT = LECTURE / "lab-submission"
sys.path.insert(0, str(ROOT / "src"))
from common import EVIDENCE, RESULTS, digest, read_json, timestamp, write_json
from salt import get_student_params
from scaling_study import stencil_reference


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def table(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def close(actual, expected, label):
    require(math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), label)


def main():
    receipt = EVIDENCE / "local_verification.json"
    receipt.unlink(missing_ok=True)
    required_sources = ["profile_env.py", "salt.py", "stride_bench.py", "dense_kernel.py",
                        "false_sharing.py", "scaling_study.py"]
    required_results = ["host_profile.json", "salted_params.json", "stride_results.tsv",
                        "loop_interchange.json", "false_sharing_karp_flatt.tsv",
                        "scaling_metrics.tsv", "roofline_summary.json"]
    require(ROOT.name == "lab-submission", "Task 4.2 requires a lab-submission directory")
    for name in required_sources:
        require((ROOT / "src" / name).is_file(), f"Missing required source: {name}")
    for name in required_results:
        require((RESULTS / name).is_file(), f"Missing required result: {name}")
    allowed_results = set(required_results + ["verification_receipt.sig"])
    require({p.name for p in RESULTS.iterdir()} == allowed_results, "Submission results must contain exactly eight required files")
    paragraphs = [p for p in (ROOT / "README.md").read_text(encoding="utf-8").strip().split("\n\n") if p.strip()]
    require(len(paragraphs) == 1, "Task 4.2 requires a one-paragraph summary README")
    settings = read_json(RESULTS / "salted_params.json")
    require(settings == get_student_params(settings["student_id"]), "Salt mismatch")
    require(settings["student_id"] == "230103341", "Wrong student ID")
    host = read_json(RESULTS / "host_profile.json")
    require(host["cpu_model"] == host["cpu"]["model"], "Instructor CPU model alias mismatch")
    require(host["physical_cores"] == host["cpu"]["physical_cores"], "Instructor physical core alias mismatch")
    require(host["logical_threads"] == host["cpu"]["logical_threads"], "Instructor logical thread alias mismatch")
    require(host["cache_line_bytes"] == 64, "Instructor cache-line alias mismatch")
    require(host["cpu"]["physical_cores"] > 0, "Missing core telemetry")
    require(host["cpu"]["logical_threads"] >= host["cpu"]["physical_cores"], "Invalid topology")
    for key in ("l1_data_bytes_per_cache", "l2_bytes_per_cache", "l3_bytes_per_cache", "cache_line_sizes_bytes"):
        require(bool(host["cache_summary"][key]), f"Missing cache telemetry: {key}")
    print("[PASS] M0: host topology and student salting")

    strides = table("stride_results.tsv")
    require([int(r["stride"]) for r in strides] == [1, 2, 4, 8, 16, 32, 64, 128], "Stride cases incomplete")
    for row in strides:
        touched = len(range(0, settings["buffer_elements"], int(row["stride"])))
        require(int(row["elements_touched"]) == touched and float(row["sum"]) == touched, "Stride sum mismatch")
        samples = [int(row[f"run_{i}_ns"]) for i in range(1, 6)]
        require(min(samples) > 0, "Invalid stride timing")
        close(row["median_time_ns"], statistics.median(samples), "Stride median mismatch")
        close(row["ns_per_element"], statistics.median(samples)/touched, "Stride normalization mismatch")
        close(row["latency_ns"], statistics.median(samples)/touched, "Instructor latency alias mismatch")
    dense = read_json(RESULTS / "loop_interchange.json")
    require(dense["N"] == settings["N"] and dense["M"] == settings["N"]//2, "Incorrect dense dimensions")
    n, m = dense["N"], dense["M"]
    rng = np.random.default_rng(settings["seed"])
    a, b = rng.random((n, n)), rng.random((n, n))
    reference = a[:m, :m] @ b[:m, :m]
    with np.load(EVIDENCE / "dense_outputs.npz", allow_pickle=False) as outputs:
        for name in ("ijk", "ikj", "reference"):
            require(outputs[name].shape == (m, m), "Invalid dense output shape")
            np.testing.assert_allclose(outputs[name], reference, rtol=0, atol=1e-12)
        np.testing.assert_allclose(outputs["ijk"], outputs["ikj"], rtol=0, atol=1e-12)
        for name in ("ijk", "ikj"):
            require(digest(outputs[name], settings["student_id"]) == dense["digest"][name], "Dense digest mismatch")
            require(len(dense["runs_ns"][name]) == 3, "Dense sample count mismatch")
            close(dense["median_ns"][name], statistics.median(dense["runs_ns"][name]), "Dense median mismatch")
    close(dense["speedup"], dense["median_ns"]["ijk"]/dense["median_ns"]["ikj"], "Dense speedup mismatch")
    close(dense["naive_time_s"], dense["median_ns"]["ijk"]/1e9, "Instructor naive time alias mismatch")
    close(dense["interchanged_time_s"], dense["median_ns"]["ikj"]/1e9, "Instructor interchanged time alias mismatch")
    print(f"[PASS] M1: eight stride cases; dense arrays within 1e-12; digest {dense['digest']['ikj'][:16]}")

    counter_runs = read_json(EVIDENCE / "counter_outputs.json")
    sharing = table("false_sharing_karp_flatt.tsv")
    require(len(counter_runs) == len(sharing), "Counter run table mismatch")
    method = read_json(EVIDENCE / "false_sharing_method.json")
    require(method["iterations_per_worker"] == 5_000_000, "Counter iteration requirement not met")
    require(len(sharing) == 3*4*method["runs"], "Missing false-sharing runs")
    for mode in ("unpadded", "padded", "privatized"):
        t1 = statistics.median(int(r["time_ns"]) for r in sharing if r["mode"] == mode and int(r["P"]) == 1)
        for p in (1, 2, 4, 8):
            group = [r for r in sharing if r["mode"] == mode and int(r["P"]) == p]
            require(len(group) == method["runs"], "Missing counter case")
            tp = statistics.median(int(r["time_ns"]) for r in group)
            for row in group:
                close(row["median_time_ns"], tp, "Counter median mismatch")
                close(row["handout_speedup"], t1/tp, "Counter speedup mismatch")
                require(int(row["base_address_mod_64"]) == 0, "Counter storage not cache aligned")
                require(int(row["total_count"]) == p*5_000_000, "Counter total mismatch")
                require(int(row["lifecycle_ns"]) >= int(row["time_ns"]) > 0, "Counter timing mismatch")
                if p > 1:
                    close(row["karp_flatt_handout"], (p*tp/t1-1)/(p-1), "Karp-Flatt mismatch")
                    close(row["work_normalized_e"], (tp/t1-1)/(p-1), "Work-normalized metric mismatch")
    for result in counter_runs:
        require(result["counts"] == [5_000_000]*result["P"], "Individual counter mismatch")
        matches = [r for r in sharing if r["mode"] == result["mode"] and int(r["P"]) == result["P"] and int(r["run"]) == result["run"]]
        require(len(matches) == 1, "Counter artifact not linked to unique timing row")
        spacing = 64 if result["mode"] == "padded" else 8
        require(result["counter_offsets_bytes"] == [i*spacing for i in range(result["P"])], "Counter layout mismatch")
    print("[PASS] M2: every counter exact; P=8 totals 40,000,000; Karp-Flatt and layouts validated")

    scaling = table("scaling_metrics.tsv")
    roofline = read_json(RESULTS / "roofline_summary.json")
    counts = roofline["process_counts"]
    require(counts[0] == 1 and counts[-1] == host["cpu"]["physical_cores"], "Scaling core counts incomplete")
    checksum_rows = read_json(EVIDENCE / "stencil_checksums.json")
    require(len(checksum_rows) == len(scaling), "Stencil checksum row count mismatch")
    for mode in ("strong", "weak"):
        t1 = statistics.median(int(r["time_ns"]) for r in scaling if r["mode"] == mode and int(r["P"]) == 1)
        for p in counts:
            group = [r for r in scaling if r["mode"] == mode and int(r["P"]) == p]
            require(len(group) == 3, "Missing scaling samples")
            n = settings["N"] if mode == "strong" else math.ceil(settings["N"]*math.sqrt(p))
            source = np.random.default_rng(settings["seed"]).random((n, n))
            reference = stencil_reference(source)
            reference_digest = digest(reference, settings["student_id"])
            if mode == "strong" and p == counts[-1]:
                saved = np.load(EVIDENCE / "stencil_output.npy", allow_pickle=False)
                require(saved.shape == (n, n), "Stencil shape mismatch")
                np.testing.assert_allclose(saved, reference, rtol=0, atol=1e-12)
            tp = statistics.median(int(r["time_ns"]) for r in group)
            for row in group:
                require(int(row["N"]) == n, "Scaling dimension mismatch")
                require(row["digest"] == reference_digest, "Stencil digest mismatch")
                require(int(row["lifecycle_ns"]) >= int(row["time_ns"]) > 0, "Stencil timing mismatch")
                close(row["median_time_ns"], tp, "Scaling median mismatch")
                close(row["checksum"], float(np.sum(reference)), "Stencil checksum mismatch")
                require(float(row["max_abs_error"]) <= 1e-12, "Stencil tolerance mismatch")
                work = (n-2)**2*int(row["repeats"])
                require(int(row["work_cells"]) == work, "Scaling workload mismatch")
                if mode == "strong":
                    close(row["speedup"], t1/tp, "Strong speedup mismatch")
                    close(row["efficiency_percent"], 100*t1/(p*tp), "Efficiency mismatch")
                    if p > 1:
                        close(row["effective_serial_fraction"], (p*tp/t1-1)/(p-1), "Serial fraction mismatch")
                else:
                    w1 = (settings["N"]-2)**2*int(row["repeats"])
                    close(row["handout_scaled_speedup"], p*w1*t1/(work*tp), "Handout weak formula mismatch")
                    close(row["conventional_scaled_speedup"], work*t1/(w1*tp), "Scaled speedup mismatch")
    for row, check in zip(scaling, checksum_rows):
        require(check["mode"] == row["mode"] and check["P"] == int(row["P"]) and check["run"] == int(row["run"]), "Checksum provenance mismatch")
        require(check["digest"] == check["reference_digest"] == row["digest"], "Stencil reference digest mismatch")
    close(roofline["naive_arithmetic_intensity_flop_per_byte"], 5/48, "Arithmetic intensity mismatch")
    close(roofline["ideal_streaming_arithmetic_intensity_flop_per_byte"], 5/16, "Streaming arithmetic intensity mismatch")
    close(roofline["arithmetic_intensity"], 5/48, "Instructor arithmetic intensity alias mismatch")
    strong_medians = {p: statistics.median(int(r["time_ns"]) for r in scaling if r["mode"] == "strong" and int(r["P"]) == p) for p in counts}
    x = np.array([1-1/p for p in counts if p > 1])
    y = np.array([strong_medians[p]/strong_medians[1]-1/p for p in counts if p > 1])
    if x.size:
        raw_fit = float(np.dot(x, y)/np.dot(x, x))
        constrained = min(1.0, max(0.0, raw_fit))
        close(roofline["amdahl_fit"]["raw_serial_fraction"], raw_fit, "Amdahl fit mismatch")
        close(roofline["amdahl_fit"]["fitted_serial_fraction"], constrained, "Constrained fit mismatch")
        if constrained:
            close(roofline["amdahl_fit"]["asymptotic_speedup_ceiling"], 1/constrained, "Amdahl ceiling mismatch")
        else:
            require(roofline["amdahl_fit"]["asymptotic_speedup_ceiling"] is None, "Unjustified finite ceiling")
    memory_peak = host.get("memory", {}).get("theoretical_peak_bandwidth_gb_s")
    if memory_peak:
        close(roofline["theoretical_peak_memory_bandwidth_gb_s"], memory_peak, "Memory peak mismatch")
        close(roofline["theoretical_streaming_roof_gflops"], memory_peak*5/16, "Roofline bound mismatch")
    for row in sharing:
        p = int(row["P"])
        if p > 1:
            for mode in ("unpadded", "padded"):
                original_row = next(r for r in sharing if r["mode"] == mode and int(r["P"]) == p)
                close(row["e_" + mode], float(original_row["karp_flatt_handout"]), "Instructor Karp-Flatt alias mismatch")
    official = dict(line.split("=", 1) for line in (RESULTS / "verification_receipt.sig").read_text(encoding="utf-8").splitlines())
    expected_token = hashlib.sha256(f"CS482-PRACTICUM-VERIFIED-{settings['student_id']}-{settings['N']}-HASH-SOLO-LAB".encode()).hexdigest()
    require(official == {"STUDENT_ID": settings["student_id"], "SALTED_N": str(settings["N"]), "SIGNATURE": expected_token}, "Instructor receipt mismatch")
    print("[PASS] M3: all 24 stencil run digests and retained output match regenerated references")

    # Plain hashes provide local reproducibility, not a secret-key signature or certification.
    files = [p for p in (ROOT / "src").iterdir() if p.is_file()]
    files += [ROOT / "verify_outputs.py", ROOT / "run_all.py", ROOT / "requirements.txt", ROOT / "README.md", Path(__file__)]
    files += [p for p in RESULTS.iterdir() if p.is_file() and p.suffix in (".json", ".tsv", ".sig")]
    files += [p for p in EVIDENCE.iterdir() if p.is_file() and p.name in ("counter_outputs.json", "false_sharing_method.json", "stencil_checksums.json", "dense_outputs.npz", "stencil_output.npy")]
    hashes = {str(p.relative_to(LECTURE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    write_json(receipt, {"status": "PASS", "generated_at": timestamp(), "student_id": settings["student_id"],
                         "verification_type": "Local numerical checks and SHA-256 file manifest; unsigned",
                         "instructor_verifier": "Full provided verify_outputs.py is run separately to produce results/verification_receipt.sig",
                         "submission_layout": "Task 4.2 required sources, results and one-paragraph README validated",
                         "rounding": "Six decimal places, C-order little-endian float64, ID + NUL prefix",
                         "files_sha256": hashes})
    print("All local numerical checks passed. Wrote evidence/local_verification.json.")
    print("[PASS] M4: lab-submission layout and one-paragraph README validated")


if __name__ == "__main__":
    main()
