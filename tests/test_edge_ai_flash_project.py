import random
import unittest

from edge_ai_flash_project import (
    AIPlacementPolicy,
    EdgeFlashModel,
    WorkloadProfile,
    generate_workloads,
    simulate_ai_optimized,
    simulate_baseline,
)


class EdgeAIFlashProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flash = EdgeFlashModel()

    def test_generate_workloads_respects_ranges(self) -> None:
        workloads = generate_workloads(count=25, seed=7)

        self.assertEqual(len(workloads), 25)
        for workload in workloads:
            self.assertGreaterEqual(workload.access_frequency, 0.0)
            self.assertLessEqual(workload.access_frequency, 1.0)
            self.assertGreaterEqual(workload.write_ratio, 0.0)
            self.assertLessEqual(workload.write_ratio, 1.0)
            self.assertGreaterEqual(workload.temporal_reuse, 0.0)
            self.assertLessEqual(workload.temporal_reuse, 1.0)
            self.assertIn(workload.block_size_kb, {4, 8, 16, 32, 64})

    def test_hot_profile_prefers_hot_cache(self) -> None:
        profile = WorkloadProfile(
            block_id=1,
            access_frequency=0.95,
            write_ratio=0.10,
            temporal_reuse=0.90,
            block_size_kb=16,
        )
        remaining = {"HOT_CACHE": 5, "BALANCED": 5, "COLD_DENSE": 5}

        zone = AIPlacementPolicy.pick_zone(profile, self.flash, remaining)

        self.assertEqual(zone, "HOT_CACHE")

    def test_optimized_strategy_improves_latency_and_energy(self) -> None:
        workloads = generate_workloads(count=180, seed=123)

        random.seed(99)
        baseline = simulate_baseline(workloads, self.flash)
        optimized = simulate_ai_optimized(workloads, self.flash)

        self.assertLess(optimized.avg_latency_ms, baseline.avg_latency_ms)
        self.assertLess(optimized.avg_energy_mj, baseline.avg_energy_mj)
        self.assertGreater(optimized.throughput_ops_per_s, 0.0)


if __name__ == "__main__":
    unittest.main()
