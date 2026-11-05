"""Verify completeness, links, source compilation, retained measurements and figures."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from fit_gustafson import fit, load_results
from models import amdahl, gustafson

ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = {
    1: "theory/task01_scaling_models.md", 2: "theory/task02_derivation.md",
    3: "theory/task03_serial_invariance.md", 4: "theory/task04_asymptotic_limits.md",
    5: "theory/task05_historical_ncube.md", 6: "calculations/task06_parametric_table.md",
    7: "calculations/task07_tolerable_serial.md", 8: "calculations/task08_cross_divergence.md",
    9: "calculations/task09_multicomponent.md", 10: "calculations/task10_memory_growth.md",
    11: "theory/task11_sun_ni_comparison.md", 12: "theory/task12_communication_penalty.md",
    13: "theory/task13_failure_cases.md", 14: "theory/task14_isoefficiency.md",
    15: "theory/task15_energy_scaling.md", 16: "calculations/task16_gemm_scaling.md",
    17: "calculations/task17_stencil_scaling.md", 18: "analysis/task18_nbody_scaling.md",
    19: "analysis/task19_monte_carlo.md", 20: "analysis/task20_graph_processing.md",
    21: "benchmarks/simulator.py", 22: "benchmarks/weak_scale_bench/benchmark.py",
    23: "benchmarks/results.csv", 24: "analysis/task24_hardware_audit.md", 25: "README.md",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_markdown(text, path):
    """Reject unsupported raw TeX in prose/math without flagging shell paths."""
    fence = None
    tex = re.compile(r"\\(?:[A-Za-z]+|[\[\]()])|\$")
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            fence = line[3:].strip() if fence is None else None
        elif fence in (None, "", "text"):
            require(not tex.search(line), f"Unreadable raw math: {path}:{line_number}")
    require(fence is None, f"Unclosed Markdown fence: {path}")


def main():
    checks = {}
    for task, name in TASK_FILES.items():
        path = ROOT / name
        require(path.is_file() and path.stat().st_size > 100, f"Task {task:02}: missing/empty {path}")
    checks["task_deliverables"] = len(TASK_FILES)
    markdowns = list(ROOT.rglob("*.md"))
    links = 0
    for path in markdowns:
        text = path.read_text(encoding="utf-8")
        check_markdown(text, path)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            require((path.parent / target).exists(), f"Broken local link: {path} -> {target}")
            links += 1
    checks["markdown_files"] = len(markdowns)
    checks["readable_markdown_math"] = True
    checks["valid_local_links"] = links
    sources = list((ROOT / "benchmarks").rglob("*.py"))
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    checks["compiled_python_sources"] = len(sources)
    summary = load_results(ROOT / "benchmarks/results.csv")
    metadata = json.loads((ROOT / "benchmarks/results.metadata.json").read_text(encoding="utf-8"))
    require(metadata["status"] == "complete", "Benchmark incomplete")
    require([r["P"] for r in summary] == metadata["workers"], "Metadata/CSV worker counts differ")
    require(all(r["k"] == metadata["k"] and r["runs"] == metadata["runs"] for r in summary), "Metadata parameters differ")
    require(len(summary) == metadata["host"]["logical_processors"], "Submission did not measure all host worker counts")
    with (ROOT / "benchmarks/results.csv").open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    require(all(r["run_id"] == metadata["run_id"] for r in raw), "Metadata session UUID differs")
    checks["validated_empirical_trials"] = sum(r["runs"] for r in summary)
    saved_fit = json.loads((ROOT / "benchmarks/fit.json").read_text(encoding="utf-8"))
    current_fit = fit(summary)
    for field in ("alpha_observed", "rmse", "r_squared"):
        require(math.isclose(saved_fit[field], current_fit[field], rel_tol=1e-12), f"Stale {field}")
    with (ROOT / "benchmarks/summary.csv").open(newline="", encoding="utf-8") as handle:
        saved_summary = list(csv.DictReader(handle))
    require(len(saved_summary) == len(summary), "Stale summary row count")
    for row, saved in zip(summary, saved_summary):
        for field, value in row.items():
            require(math.isclose(value, float(saved[field]), rel_tol=1e-12, abs_tol=1e-15), f"Stale summary {field}")
    checks["saved_statistics_recomputed"] = True
    with (ROOT / "calculations/simulated_scaling.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    for row in predictions:
        p, alpha = int(row["P"]), float(row["fraction"])
        require(1 <= p <= 65536 and .001 <= alpha <= .20, "Simulator range error")
        for key, expected in (("amdahl_speedup", amdahl(p, alpha)),
                              ("gustafson_speedup", gustafson(p, alpha)),
                              ("amdahl_efficiency", amdahl(p, alpha)/p),
                              ("gustafson_efficiency", gustafson(p, alpha)/p)):
            require(math.isclose(float(row[key]), expected, rel_tol=1e-12), f"Bad prediction: {key}")
    require(len(predictions) == 68, "Expected committed default 17 core counts x 4 fractions")
    checks["validated_simulator_rows"] = len(predictions)
    for stem in ("scaling_comparison", "model_overlay"):
        png, pdf = ROOT / f"plots/{stem}.png", ROOT / f"plots/{stem}.pdf"
        data = png.read_bytes()
        require(data[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG {png}")
        width, height = struct.unpack(">II", data[16:24])
        require(width >= 3000 and height >= 1200, f"Low resolution {png}")
        require(pdf.read_bytes().startswith(b"%PDF-"), f"Invalid PDF {pdf}")
        checks[f"{stem}_png_pixels"] = [width, height]
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "verification.json":
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(verified_at=datetime.now().astimezone().isoformat(), student_id="230103341",
                  status="passed", checks=checks, sha256=hashes)
    output = ROOT / "benchmarks/logs/verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()