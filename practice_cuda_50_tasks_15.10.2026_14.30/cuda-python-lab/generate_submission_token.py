"""Trusted-operator HMAC utility; not an instructor-issued grading credential.

LAB_HMAC_SECRET must be supplied by a trusted operator, never committed or placed
in a student-controlled notebook. A client-held HMAC secret cannot prevent fraud.
The hardware metadata is signed context, not cryptographic hardware attestation.
"""
import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path

from lab_support import ROOT, source_manifest


def validate_report(report):
    if (report.get("mode") != "real_gpu" or not report.get("certified")
            or report.get("module") is not None
            or report.get("pytest_exit_code") != 0
            or report.get("passed_task_numbers") != list(range(1, 51))
            or report.get("counts", {}).get("skipped", 1)
            or report.get("counts", {}).get("failed", 1)):
        raise ValueError("Signing requires a complete real-GPU run, with no skips or failures")
    if report.get("source_manifest") != source_manifest():
        raise ValueError("Source changed since grading; rerun the full GPU suite")
    if not report.get("environment", {}).get("nvidia_smi"):
        raise ValueError("GPU identity metadata is missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_id")
    parser.add_argument("student_name")
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "submission_token.json")
    args = parser.parse_args(argv)
    secret = os.environ.get("LAB_HMAC_SECRET", "")
    if len(secret.encode()) < 32:
        parser.error("Trusted operator must provide LAB_HMAC_SECRET (at least 32 bytes); no default key exists")
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        parser.error("Cannot sign simulator results")
    report_path = ROOT / "result" / "signing_run.json"
    subprocess.run([sys.executable, str(ROOT / "autograder.py"), "--output", str(report_path)],
                   cwd=ROOT, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    payload = {"student_id": args.student_id, "student_name": args.student_name,
               "report": report, "issuer": "local_trusted_operator_not_official_instructor"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    document = {"algorithm": "HMAC-SHA256", "payload": payload,
                "signature": hmac.new(secret.encode(), encoded, hashlib.sha256).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Signed full GPU report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())