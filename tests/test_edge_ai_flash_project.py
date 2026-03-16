import random
import unittest
from collections import Counter

from edge_ai_flash_project import (
    AIPlacementPolicy,
    EdgeFlashModel,
    WorkloadProfile,
    build_workloads_from_live_activity,
    generate_workloads,
    get_last_simulation_diagnostics,
    run_simulation,
    simulate_ai_optimized,
    simulate_baseline,
)
from ml.inference import load_trained_model, predict_profile_zone
from ml.training import generate_labeled_workloads, train_and_save_model


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

    def test_generate_labeled_workloads_returns_matching_lengths(self) -> None:
        workloads, labels = generate_labeled_workloads(sample_count=200, seed=99)
        label_counts = Counter(labels)

        self.assertEqual(len(workloads), 200)
        self.assertEqual(len(labels), 200)
        self.assertTrue(set(labels).issubset({"HOT_CACHE", "BALANCED", "COLD_DENSE"}))
        self.assertGreaterEqual(len(label_counts), 3)
        for zone_name in {"HOT_CACHE", "BALANCED", "COLD_DENSE"}:
            self.assertGreater(label_counts[zone_name], 0)

    def test_ml_artifact_can_predict_zone(self) -> None:
        artifacts = train_and_save_model(sample_count=1200, seed=2026)
        artifact = load_trained_model()
        profile = WorkloadProfile(
            block_id=99,
            access_frequency=0.91,
            write_ratio=0.08,
            temporal_reuse=0.87,
            block_size_kb=16,
        )

        prediction = predict_profile_zone(profile, artifact)

        self.assertIn(prediction.zone, {"HOT_CACHE", "BALANCED", "COLD_DENSE"})
        self.assertGreaterEqual(prediction.priority_score, 0.0)
        self.assertIn("accuracy", artifacts.evaluation)
        self.assertIn("macro_f1", artifacts.evaluation)
        self.assertIn("per_class", artifacts.evaluation)
        self.assertIn("confusion_matrix", artifacts.evaluation)
        self.assertGreaterEqual(artifacts.evaluation["accuracy"], 0.0)
        self.assertLessEqual(artifacts.evaluation["accuracy"], 1.0)
        self.assertGreater(artifacts.evaluation["per_class"]["BALANCED"]["support"], 0)
        self.assertGreater(artifacts.evaluation["per_class"]["COLD_DENSE"]["support"], 0)

    def test_run_simulation_supports_ml_policy(self) -> None:
        train_and_save_model(sample_count=1200, seed=2026)

        baseline, optimized, total_blocks = run_simulation(count=90, seed=11, policy_mode="ml")

        self.assertEqual(total_blocks, 90)
        self.assertGreater(baseline.avg_latency_ms, 0.0)
        self.assertGreater(optimized.avg_latency_ms, 0.0)

    def test_overflow_is_tracked_when_workload_exceeds_capacity(self) -> None:
        baseline, optimized, total_blocks = run_simulation(count=500, seed=123)
        diagnostics = get_last_simulation_diagnostics()

        self.assertEqual(total_blocks, 500)
        self.assertGreater(baseline.avg_latency_ms, 0.0)
        self.assertGreater(optimized.avg_latency_ms, 0.0)
        self.assertEqual(diagnostics["total_capacity"], 250)
        self.assertEqual(diagnostics["optimized"]["overflow_count"], 250)
        self.assertEqual(diagnostics["baseline"]["overflow_count"], 250)

    def test_optimized_overflow_spreads_across_non_cold_zones(self) -> None:
        run_simulation(count=500, seed=321)
        diagnostics = get_last_simulation_diagnostics()
        overflow_by_zone = diagnostics["optimized"]["overflow_by_zone"]

        self.assertGreater(sum(overflow_by_zone.values()), 0)
        self.assertGreater(overflow_by_zone["HOT_CACHE"] + overflow_by_zone["BALANCED"], 0)


if __name__ == "__main__":
    unittest.main()
