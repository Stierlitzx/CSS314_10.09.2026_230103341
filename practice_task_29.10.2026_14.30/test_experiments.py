"""Small correctness tests, including real workers and uneven partitions."""
import unittest

from benchmark_support import partition, scaling_metrics
from exp_a_serial import measure_experiment
from exp_b_contention import measure_contended
from exp_c_fixed import measure_privatized


class ExperimentTests(unittest.TestCase):
    def test_partition_preserves_remainder(self):
        self.assertEqual(partition(11, 3), [4, 4, 3])
        self.assertEqual(partition(2, 4), [1, 1, 0, 0])
        self.assertEqual(partition(0, 2), [0, 0])

    def test_invalid_workloads(self):
        for total, workers in ((-1, 2), (1, 0)):
            with self.assertRaises(ValueError):
                partition(total, workers)
        with self.assertRaises(ValueError):
            measure_contended(1, 2_147_483_648)

    def test_known_amdahl_metrics(self):
        for workers, runtime in ((2, 1.8), (4, 1.1)):
            self.assertAlmostEqual(scaling_metrics(3.2, runtime, workers)["karp_flatt"], 0.125)
        self.assertEqual(scaling_metrics(1, 1, 1)["karp_flatt"], None)
        self.assertAlmostEqual(scaling_metrics(4, 1, 4)["efficiency_percent"], 100)

    def test_negative_speedup_is_not_clamped(self):
        result = scaling_metrics(0.65, 2.4, 4)
        self.assertLess(result["speedup"], 1)
        self.assertAlmostEqual(result["karp_flatt"], 4.58974358974359)

    def test_real_serial_tail_and_pool_results(self):
        result = measure_experiment(2, total_chunks=3, chunk_size=51)
        self.assertTrue(result["correct"])
        self.assertEqual(len(result["chunk_values"]), 3)
        self.assertGreaterEqual(result["actual_pause_ns"], 400_000_000)

    def test_contended_workers_keep_uneven_total(self):
        result = measure_contended(3, total_increments=101)
        self.assertTrue(result["correct"])
        self.assertEqual(result["actual_total"], 101)
        self.assertEqual(result["assigned_increments"], [34, 34, 33])

    def test_private_workers_keep_uneven_total(self):
        result = measure_privatized(3, total_increments=101)
        self.assertTrue(result["correct"])
        self.assertEqual(result["partial_totals"], [34, 34, 33])


if __name__ == "__main__":
    unittest.main(verbosity=2)
