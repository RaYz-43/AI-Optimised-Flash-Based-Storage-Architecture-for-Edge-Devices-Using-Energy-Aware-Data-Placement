import random
import unittest

from edge_ai_flash_project import (
    AIPlacementPolicy,
    EdgeFlashModel,
    WorkloadProfile,
    build_workloads_from_live_activity,
    generate_workloads,
    run_simulation,
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

    def test_run_simulation_returns_expected_block_count(self) -> None:
        baseline, optimized, total_blocks = run_simulation(count=120, seed=55)

        self.assertEqual(total_blocks, 120)
        self.assertGreater(baseline.avg_latency_ms, 0.0)
        self.assertGreater(optimized.avg_latency_ms, 0.0)

    def test_build_workloads_from_live_activity_maps_process_rows(self) -> None:
        workloads = build_workloads_from_live_activity(
            [
                {"pid": 1010, "total_ops": 20, "write_ops": 8, "total_bytes": 131072, "active_samples": 4},
                {"pid": 2020, "total_ops": 10, "write_ops": 1, "total_bytes": 32768, "active_samples": 2},
            ],
            sample_count=4,
        )

        self.assertEqual(len(workloads), 2)
        self.assertEqual(workloads[0].block_id, 1010)
        self.assertGreaterEqual(workloads[0].access_frequency, workloads[1].access_frequency)
        self.assertGreater(workloads[0].write_ratio, workloads[1].write_ratio)


if __name__ == "__main__":
    unittest.main()
