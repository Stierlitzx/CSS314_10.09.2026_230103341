"""Deterministic model/data tests and a small real multiprocessing smoke test."""
import csv
from contextlib import redirect_stdout
import io
import math
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import report_results
from fit_gustafson import fit, load_results
from models import amdahl, equivalent_serial_fraction, gustafson
from simulator import parse_cores, parse_fractions, simulate
from weak_scale_bench import benchmark
from verify_artifacts import check_markdown

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_endpoints(self):
        for p in (1, 16, 65536):
            self.assertEqual(gustafson(p, 0), p)
            self.assertEqual(gustafson(p, 1), 1)
            self.assertEqual(amdahl(p, 0), p)
            self.assertEqual(amdahl(p, 1), 1)

    def test_reference_conversion(self):
        for p in (1, 2, 128, 1024, 65536):
            for alpha in (0, 0.001, 0.08, 0.2, 1):
                s = equivalent_serial_fraction(p, alpha)
                self.assertAlmostEqual(amdahl(p, s), gustafson(p, alpha), places=9)

    def test_known_calculations(self):
        self.assertAlmostEqual(gustafson(16, 0.04), 15.4)
        self.assertAlmostEqual(gustafson(16384, 0.04), 15728.68)
        self.assertAlmostEqual(gustafson(8192, 692/8191), 7500)
        self.assertAlmostEqual(amdahl(128, 0.08), 3200/279)
        self.assertAlmostEqual(gustafson(128, 0.08), 117.84)

    def test_invalid_models(self):
        for p, a in ((0, .1), (-1, .1), (2.5, .1), (True, .1), (2, -1), (2, 1.1),
                     (2, math.nan), (2, math.inf)):
            with self.subTest(p=p, a=a), self.assertRaises(ValueError):
                gustafson(p, a)

    def test_cli_ranges(self):
        import argparse
        self.assertEqual(parse_cores("65536,1,1,7"), [1, 7, 65536])
        self.assertEqual(parse_fractions("0.2,0.001"), [0.001, 0.2])
        for value in ("0", "65537", "", "1.5"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_cores(value)
        for value in ("0", "0.201", "nan", "inf", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_fractions(value)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "subfolder" / "prediction.csv"
            rows = simulate([1, 7, 65536], [0.001, 0.2], path)
            with path.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 6)
            self.assertEqual(len(rows), 6)
            self.assertEqual(float(saved[0]["gustafson_speedup"]), 1)


class RegressionTests(unittest.TestCase):
    def test_exact_recovery(self):
        data = [dict(P=p, mean_scaled_speedup=gustafson(p, .08)) for p in (1, 2, 4, 8, 12)]
        fitted = fit(data)
        self.assertAlmostEqual(fitted["alpha_observed"], .08)
        self.assertAlmostEqual(fitted["rmse"], 0)
        self.assertAlmostEqual(fitted["r_squared"], 1)
        self.assertAlmostEqual(fitted["free_intercept_diagnostic"]["intercept"], .08)

    def test_does_not_clamp_nonphysical_fit(self):
        data = [dict(P=p, mean_scaled_speedup=2*p-1) for p in (1, 2, 4, 8)]
        fitted = fit(data)
        self.assertAlmostEqual(fitted["alpha_observed"], -1)
        self.assertFalse(fitted["alpha_in_physical_range"])

    def test_fit_identifiability(self):
        with self.assertRaises(ValueError):
            fit([dict(P=1, mean_scaled_speedup=1)])
        for p in (0, -1, True, 2.5):
            with self.subTest(p=p), self.assertRaises(ValueError):
                fit([dict(P=1, mean_scaled_speedup=1), dict(P=p, mean_scaled_speedup=2)])
        with self.assertRaises(ValueError):
            fit([dict(P=2, mean_scaled_speedup=1), dict(P=2, mean_scaled_speedup=2)])
        for speed in (math.nan, math.inf, -math.inf):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                fit([dict(P=1, mean_scaled_speedup=1), dict(P=2, mean_scaled_speedup=speed)])

    def dataset(self, path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=benchmark.FIELDS)
            writer.writeheader()
            for p in range(1, 6):
                for run in range(1, 6):
                    speed = gustafson(p, .1)
                    writer.writerow(dict(run_id="synthetic-test-only", timestamp="2026-11-05T00:00:00Z",
                                         P=p, run=run, k=100000, total_points=p*100000,
                                         serial_seconds=speed, parallel_seconds=1,
                                         scaled_speedup=speed, serial_integral=math.pi,
                                         parallel_integral=math.pi, absolute_pi_error=0,
                                         worker_pids=";".join(str(i+10) for i in range(p)),
                                         measurement_order="serial-first", correct=True))

    def test_statistics_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.csv"
            self.dataset(path)
            summary = load_results(path)
            self.assertEqual(len(summary), 5)
            self.assertEqual(summary[-1]["runs"], 5)
            self.assertEqual(summary[-1]["se_scaled_speedup"], 0)
            self.assertAlmostEqual(fit(summary)["alpha_observed"], .1)

    def test_rejects_corrupt_data(self):
        mutations = [dict(correct="False"), dict(total_points="2"), dict(parallel_seconds="nan"),
                     dict(scaled_speedup="999"), dict(worker_pids="10;10"), dict(k="100001"),
                     dict(parallel_integral="2.0"), dict(absolute_pi_error="0.2"),
                     dict(worker_pids="10;010")]
        for change in mutations:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "test.csv"
                self.dataset(path)
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                target = 5 if change.get("worker_pids") == "10;010" else 0
                rows[target].update(change)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=benchmark.FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_results(path)

    def test_rejects_duplicate_missing_and_empty(self):
        for kind in ("duplicate", "missing", "empty"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "test.csv"
                self.dataset(path)
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                rows = rows + [rows[0]] if kind == "duplicate" else rows[1:] if kind == "missing" else []
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=benchmark.FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_results(path)


class DocumentationTests(unittest.TestCase):
    def test_markdown_math_is_readable(self):
        for path in ROOT.rglob("*.md"):
            with self.subTest(path=path.relative_to(ROOT)):
                check_markdown(path.read_text(encoding="utf-8"), path)

    def test_rejects_raw_tex_and_unclosed_fences(self):
        for text in (r"\[ S=\frac{1}{P} \]", "$S=P$", "```text\nS = P"):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                check_markdown(text, "example.md")
        check_markdown("```powershell\n$root = 'C:\\practice'\n```", "example.md")
        check_markdown("```text\nS = P − α × (P − 1)\n```", "example.md")

    def test_report_regenerates_submission_style(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "benchmarks").mkdir()
            for name in ("results.csv", "results.metadata.json"):
                (root / "benchmarks" / name).write_bytes((ROOT / "benchmarks" / name).read_bytes())
            with patch.object(report_results, "ROOT", root), redirect_stdout(io.StringIO()):
                report_results.main()
            relative = Path("analysis/task23_experimental_results.md")
            report = (root / relative).read_text(encoding="utf-8")
            self.assertEqual(report, (ROOT / relative).read_text(encoding="utf-8"))
            self.assertIn("I measured 60 real trials", report)
            self.assertIn("effective overhead parameter", report)
            self.assertIn("0.784991225892", report)
            check_markdown(report, relative)


class WorkloadTests(unittest.TestCase):
    def test_chunks_cover_exactly_the_scaled_domain(self):
        p, k = 5, 2000
        chunked = math.fsum(benchmark.integrate(i*k, (i+1)*k, p*k) for i in range(p))
        self.assertAlmostEqual(chunked, benchmark.integrate(0, p*k, p*k), places=12)
        self.assertLess(abs(chunked-math.pi), 2/(p*k)**2)

    def test_real_spawn(self):
        p = min(2, os.cpu_count() or 1)
        ts, serial = benchmark.serial_reference(p, 2000)
        tp, parallel, pids = benchmark.parallel_measure(p, 2000)
        self.assertGreater(ts, 0)
        self.assertGreater(tp, 0)
        self.assertEqual(len(set(pids)), p)
        self.assertAlmostEqual(serial, parallel, places=12)

    @unittest.skipUnless((ROOT / "benchmarks" / "results.csv").exists(), "No empirical data yet")
    def test_retained_empirical_dataset(self):
        summary = load_results(ROOT / "benchmarks" / "results.csv")
        self.assertGreaterEqual(len(summary), 5)
        self.assertTrue(all(row["runs"] >= 5 for row in summary))


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))