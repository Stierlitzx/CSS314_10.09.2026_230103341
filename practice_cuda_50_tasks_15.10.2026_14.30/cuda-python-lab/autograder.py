"""Reference-test runner. Simulation is useful debugging, never GPU certification."""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Results:
    def __init__(self):
        self.outcomes = {}
        self.collection_errors = []

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed or report.skipped:
            current = self.outcomes.get(report.nodeid)
            if current and current["outcome"] == "failed":
                return
            self.outcomes[report.nodeid] = {
                "outcome": report.outcome,
                "duration_s": report.duration,
                "detail": str(report.longrepr) if report.longrepr else "",
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", action="store_true", help="CPU simulation; no certification")
    parser.add_argument("--module", type=int, choices=range(1, 7))
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "autograder_report.json")
    args = parser.parse_args(argv)
    if args.simulator:
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    simulated = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
    import pytest
    from lab_support import environment, source_manifest

    results = Results()
    target = ROOT / "tests"
    if args.module:
        target /= f"test_module{args.module}.py"
    rc = int(pytest.main([str(target), "-q", "-c", str(ROOT / "pytest.ini")], plugins=[results]))
    counts = {key: sum(v["outcome"] == key for v in results.outcomes.values())
              for key in ("passed", "failed", "skipped")}
    task_numbers = sorted({
        int(match.group(1)) for node, result in results.outcomes.items()
        if result["outcome"] == "passed"
        for match in [re.search(r"test_task_?(\d+)", node)] if match
    })
    complete = (rc == 0 and not simulated and args.module is None
                and not counts["skipped"] and not counts["failed"]
                and task_numbers == list(range(1, 51)))
    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "simulator" if simulated else "real_gpu",
        "certified": complete,
        "status": "passed_gpu_reference_suite" if complete else (
            "simulated_not_certified" if simulated and rc == 0 else "incomplete_or_failed"),
        "module": args.module, "pytest_exit_code": rc,
        "counts": counts, "passed_task_numbers": task_numbers,
        "collection_errors": results.collection_errors,
        "environment": environment(), "source_manifest": source_manifest(),
        "tests": results.outcomes,
        "scope": "Functional reference tests, not the instructor's profiler or official grader.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nStatus: {payload['status']}; report: {args.output.resolve()}")
    if rc:
        return rc
    if not simulated and (counts["skipped"] or not results.outcomes):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())