import argparse
import subprocess
import sys
from pathlib import Path


PHASES = (
    "task0_cal.py",
    "task1_falsesharing.py",
    "task2_scaling.py",
    "task3_roofline.py",
    "task4_scheduling.py",
)


def main():
    parser = argparse.ArgumentParser(description="Run the five parallel-computing lab phases.")
    parser.add_argument("--quick", action="store_true", help="run validation-sized workloads")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for phase in PHASES:
        command = [sys.executable, str(root / phase)]
        if args.quick and phase != PHASES[0]:
            command.append("--quick")
        print(f"\n=== {phase} ===", flush=True)
        subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
