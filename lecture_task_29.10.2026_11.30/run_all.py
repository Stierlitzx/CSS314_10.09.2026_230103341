"""Run the complete practicum from any current directory; preserve real logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from common import RESULTS, build_native, timestamp, write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", default="230103341")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "verification_receipt.sig").unlink(missing_ok=True)
    started = time.perf_counter()
    manifest = {"started_at": timestamp(), "student_id": args.student_id,
                "scheduled_lecture": "2026-10-29 11:30", "stages": [],
                "note": "Scheduled lecture is a label; captured timestamps are actual execution time. No Git timestamps are synthesized."}
    stages = [("M0", "src/profile_env.py", "--student-id", args.student_id),
              ("M1-stride", "src/stride_bench.py"),
              ("M1-dense", "src/dense_kernel.py"),
              ("M2", "src/false_sharing.py", "--backend", "native"),
              ("M3", "src/scaling_study.py"),
              ("M4", "verify_outputs.py")]
    with (RESULTS / "run.log").open("w", encoding="utf-8") as log:
        try:
            target = build_native()
            print(f"Built native kernels: {target}", flush=True)
            for label, script, *arguments in stages:
                command = [sys.executable, "-u", str(ROOT / script), *arguments]
                stage = {"label": label, "command": command, "started_at": timestamp()}
                print(f"\n{label}: {script}", flush=True)
                log.write(f"\n{label}: {command}\n")
                log.flush()
                begin = time.perf_counter()
                environment = dict(os.environ, PYTHONIOENCODING="utf-8")
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, env=environment,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                process.wait()
                stage.update(returncode=process.returncode, elapsed_seconds=time.perf_counter()-begin, finished_at=timestamp())
                manifest["stages"].append(stage)
                write_json(RESULTS / "run_manifest.json", manifest)
                if process.returncode:
                    raise RuntimeError(f"{label} failed with exit code {process.returncode}")
            manifest["status"] = "PASS"
        except BaseException:
            manifest["status"] = "FAILED"
            raise
        finally:
            manifest.update(finished_at=timestamp(), elapsed_seconds=time.perf_counter()-started)
            write_json(RESULTS / "run_manifest.json", manifest)
    print(f"\nComplete: {manifest['elapsed_seconds']:.1f} seconds; results saved in {RESULTS}")


if __name__ == "__main__":
    main()
