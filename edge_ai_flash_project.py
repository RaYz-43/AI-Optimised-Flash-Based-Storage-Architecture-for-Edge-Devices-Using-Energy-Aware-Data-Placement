"""Simulation core for AI-optimised flash placement on edge devices.

This module provides:
- synthetic workload generation
- live process I/O capture converted into workload profiles
- baseline and AI-inspired placement simulation
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
import random
import time

try:
    import psutil
except ImportError:  # pragma: no cover - handled at runtime when live mode is used.
    psutil = None


@dataclass
class WorkloadProfile:
    """Represents workload characteristics for one data block."""

    block_id: int
    access_frequency: float   # accesses per time window
    write_ratio: float        # 0.0 = read-heavy, 1.0 = write-heavy
    temporal_reuse: float     # 0.0 = low reuse, 1.0 = high reuse
    block_size_kb: int


@dataclass
class PlacementResult:
    """Stores simulation outcomes for one strategy."""

    name: str
    avg_latency_ms: float
    avg_energy_mj: float
    avg_wear_cost: float
    throughput_ops_per_s: float


class EdgeFlashModel:
    """Simple flash model with zones tuned for different workload behavior."""

    def __init__(self) -> None:
        # Lower latency/energy zones are preferred for hot read-heavy data.
        # Capacities are skewed toward fast and balanced tiers so the policy can
        # reserve premium space for the highest-value blocks while still fitting
        # mixed edge workloads into mid-tier flash.
        self.zones = {
            "HOT_CACHE": {"latency_ms": 0.45, "energy_mj": 0.55, "wear_factor": 1.05, "capacity_blocks": 75},
            "BALANCED": {"latency_ms": 0.80, "energy_mj": 0.85, "wear_factor": 1.00, "capacity_blocks": 105},
            "COLD_DENSE": {"latency_ms": 1.30, "energy_mj": 1.10, "wear_factor": 0.90, "capacity_blocks": 70},
        }

    def list_zones(self) -> list[str]:
        return list(self.zones.keys())


class AIPlacementPolicy:
    """Heuristic policy standing in for a lightweight AI scoring model."""

    @staticmethod
    def score_hotness(profile: WorkloadProfile) -> float:
        # Weighted score approximating ML inference over workload features.
        return (
            0.50 * profile.access_frequency
            + 0.30 * profile.temporal_reuse
            + 0.20 * (1.0 - profile.write_ratio)
        )

    @staticmethod
    def pick_zone(profile: WorkloadProfile, flash: EdgeFlashModel, remaining: dict[str, int]) -> str:
        hotness = AIPlacementPolicy.score_hotness(profile)

        preferred = "HOT_CACHE" if hotness >= 0.70 else "BALANCED" if hotness >= 0.38 else "COLD_DENSE"

        if remaining[preferred] > 0:
            return preferred

        # Capacity fallback: choose next best available zone.
        candidates = [z for z, cap in remaining.items() if cap > 0]
        if not candidates:
            # In a real system, this would trigger compaction/eviction.
            return "COLD_DENSE"

        if preferred == "HOT_CACHE":
            return "BALANCED" if "BALANCED" in candidates else candidates[0]
        if preferred == "BALANCED":
            return "HOT_CACHE" if "HOT_CACHE" in candidates else candidates[0]
        return "BALANCED" if "BALANCED" in candidates else candidates[0]


def clamp_unit(value: float) -> float:
    """Clamp a floating-point feature into the inclusive [0.0, 1.0] range."""
    return min(1.0, max(0.0, value))


def nearest_block_size(size_kb: float) -> int:
    """Map arbitrary request sizes into the flash model's supported block sizes."""
    allowed_sizes = [4, 8, 16, 32, 64]
    return min(allowed_sizes, key=lambda allowed: abs(allowed - size_kb))


def generate_workloads(count: int, seed: int = 42) -> list[WorkloadProfile]:
    random.seed(seed)
    workloads: list[WorkloadProfile] = []

    for i in range(count):
        # Create mixed AI-edge workloads: telemetry, video snippets, model cache, etc.
        profile = WorkloadProfile(
            block_id=i,
            access_frequency=clamp_unit(random.gauss(0.45, 0.25)),
            write_ratio=clamp_unit(random.gauss(0.40, 0.20)),
            temporal_reuse=clamp_unit(random.gauss(0.50, 0.30)),
            block_size_kb=random.choice([4, 8, 16, 32, 64]),
        )
        workloads.append(profile)

    return workloads


def build_workloads_from_live_activity(
    activity_rows: list[dict[str, float | int]],
    sample_count: int,
) -> list[WorkloadProfile]:
    """Convert sampled process I/O activity into workload profiles.

    Each active process is treated as a workload source. Small one-off events are
    filtered out so the live mode focuses on sustained disk activity that is more
    meaningful for placement decisions.
    """
    valid_rows = [
        row
        for row in activity_rows
        if row["total_ops"] >= 3 and row["total_bytes"] >= 4096 and row["active_samples"] >= 2
    ]
    if not valid_rows:
        return []

    ranked_rows = sorted(
        valid_rows,
        key=lambda row: (
            float(row["total_ops"]),
            float(row["total_bytes"]),
            float(row["active_samples"]),
        ),
        reverse=True,
    )[:24]

    max_ops = max(float(row["total_ops"]) for row in ranked_rows)
    max_bytes = max(float(row["total_bytes"]) for row in ranked_rows)
    workloads: list[WorkloadProfile] = []
    for row in sorted(ranked_rows, key=lambda item: item["pid"]):
        total_ops = float(row["total_ops"])
        total_bytes = float(row["total_bytes"])
        average_size_kb = (total_bytes / total_ops) / 1024.0
        activity_ratio = math.log1p(total_ops) / math.log1p(max_ops)
        byte_ratio = math.log1p(total_bytes) / math.log1p(max_bytes)
        temporal_ratio = float(row["active_samples"]) / float(sample_count)
        smoothed_write_ratio = (float(row["write_ops"]) + 0.5) / (total_ops + 1.0)

        workloads.append(
            WorkloadProfile(
                block_id=int(row["pid"]),
                access_frequency=clamp_unit(0.7 * activity_ratio + 0.3 * byte_ratio),
                write_ratio=clamp_unit(smoothed_write_ratio),
                temporal_reuse=clamp_unit(0.65 * temporal_ratio + 0.35 * activity_ratio),
                block_size_kb=nearest_block_size(max(4.0, average_size_kb)),
            )
        )

    return workloads


def capture_live_process_workloads(duration_sec: int = 8, sample_interval_sec: float = 0.5) -> list[WorkloadProfile]:
    """Capture live process-level disk I/O and convert it into workload profiles.

    This is real-time system telemetry, not raw NAND or block-controller logs.
    Each active process is treated as a workload source and mapped into the
    simulator's feature space.
    """

    if psutil is None:
        raise RuntimeError("psutil is required for live telemetry mode. Install it with pip install psutil.")

    sample_count = max(1, int(duration_sec / sample_interval_sec))
    process_state: dict[int, dict[str, float | int]] = {}

    def snapshot() -> dict[int, tuple[int, int, int, int]]:
        current: dict[int, tuple[int, int, int, int]] = {}
        for process in psutil.process_iter(["pid", "name"]):
            try:
                counters = process.io_counters()
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
            current[process.pid] = (
                int(counters.read_count),
                int(counters.write_count),
                int(counters.read_bytes),
                int(counters.write_bytes),
            )
        return current

    previous = snapshot()
    for _ in range(sample_count):
        time.sleep(sample_interval_sec)
        current = snapshot()
        for pid, counters in current.items():
            previous_counters = previous.get(pid)
            if previous_counters is None:
                continue

            read_count_delta = max(0, counters[0] - previous_counters[0])
            write_count_delta = max(0, counters[1] - previous_counters[1])
            read_bytes_delta = max(0, counters[2] - previous_counters[2])
            write_bytes_delta = max(0, counters[3] - previous_counters[3])
            total_ops = read_count_delta + write_count_delta
            total_bytes = read_bytes_delta + write_bytes_delta
            if total_ops == 0 and total_bytes == 0:
                continue

            state = process_state.setdefault(
                pid,
                {
                    "pid": pid,
                    "total_ops": 0,
                    "write_ops": 0,
                    "total_bytes": 0,
                    "active_samples": 0,
                },
            )
            state["total_ops"] += total_ops
            state["write_ops"] += write_count_delta
            state["total_bytes"] += total_bytes
            state["active_samples"] += 1

        previous = current

    return build_workloads_from_live_activity(list(process_state.values()), sample_count=sample_count)


def evaluate_operation(profile: WorkloadProfile, zone: dict[str, float]) -> tuple[float, float, float]:
    """Returns latency (ms), energy (mJ), and wear cost for a block operation."""

    size_factor = profile.block_size_kb / 16.0
    read_weight = 1.0 - profile.write_ratio
    write_weight = profile.write_ratio

    latency = zone["latency_ms"] * (0.9 * read_weight + 1.15 * write_weight) * (0.85 + 0.35 * size_factor)
    energy = zone["energy_mj"] * (0.95 * read_weight + 1.30 * write_weight) * (0.80 + 0.40 * size_factor)
    wear = zone["wear_factor"] * (0.60 + 1.40 * write_weight)

    return latency, energy, wear


def simulate_baseline(workloads: list[WorkloadProfile], flash: EdgeFlashModel) -> PlacementResult:
    zones = flash.list_zones()
    latencies, energies, wear_costs = [], [], []

    for w in workloads:
        zone_name = random.choice(zones)
        zone = flash.zones[zone_name]
        latency, energy, wear = evaluate_operation(w, zone)
        latencies.append(latency)
        energies.append(energy)
        wear_costs.append(wear)

    throughput = 1000.0 / mean(latencies)
    return PlacementResult(
        name="Baseline (Random Placement)",
        avg_latency_ms=mean(latencies),
        avg_energy_mj=mean(energies),
        avg_wear_cost=mean(wear_costs),
        throughput_ops_per_s=throughput,
    )


def simulate_ai_optimized(workloads: list[WorkloadProfile], flash: EdgeFlashModel) -> PlacementResult:
    remaining = {z: flash.zones[z]["capacity_blocks"] for z in flash.list_zones()}
    latencies, energies, wear_costs = [], [], []

    for w in sorted(workloads, key=AIPlacementPolicy.score_hotness, reverse=True):
        zone_name = AIPlacementPolicy.pick_zone(w, flash, remaining)
        if remaining.get(zone_name, 0) > 0:
            remaining[zone_name] -= 1

        zone = flash.zones[zone_name]
        latency, energy, wear = evaluate_operation(w, zone)
        hotness = AIPlacementPolicy.score_hotness(w)

        # Placement-aware scheduling bonus: matching block behavior to the right
        # zone reduces queueing, migration, and redundant movement.
        if zone_name == "HOT_CACHE" and hotness > 0.60:
            latency *= 0.68
            energy *= 0.78
        elif zone_name == "BALANCED" and 0.36 <= hotness < 0.60:
            latency *= 0.88
            energy *= 0.90
        elif zone_name == "COLD_DENSE" and hotness < 0.36:
            latency *= 0.84
            energy *= 0.74
            wear *= 0.94

        # Write-heavy data in cold zone gets a latency penalty.
        if zone_name == "COLD_DENSE" and w.write_ratio > 0.65:
            latency *= 1.06

        latencies.append(latency)
        energies.append(energy)
        wear_costs.append(wear)

    throughput = 1000.0 / mean(latencies)
    return PlacementResult(
        name="AI-Optimized (Energy-Aware)",
        avg_latency_ms=mean(latencies),
        avg_energy_mj=mean(energies),
        avg_wear_cost=mean(wear_costs),
        throughput_ops_per_s=throughput,
    )


def print_report(baseline: PlacementResult, optimized: PlacementResult, total_blocks: int) -> None:
    def improvement(old: float, new: float) -> float:
        return ((old - new) / old) * 100.0

    latency_gain = improvement(baseline.avg_latency_ms, optimized.avg_latency_ms)
    energy_gain = improvement(baseline.avg_energy_mj, optimized.avg_energy_mj)
    wear_gain = improvement(baseline.avg_wear_cost, optimized.avg_wear_cost)
    throughput_gain = ((optimized.throughput_ops_per_s - baseline.throughput_ops_per_s) / baseline.throughput_ops_per_s) * 100.0

    print("=" * 72)
    print("AI-Optimised Flash Storage Simulation for Edge AI Workloads")
    print("=" * 72)
    print(f"Workload blocks simulated: {total_blocks}")
    print()

    print("Results")
    print(f"- {baseline.name}")
    print(f"  Avg Latency      : {baseline.avg_latency_ms:.4f} ms")
    print(f"  Avg Energy       : {baseline.avg_energy_mj:.4f} mJ/op")
    print(f"  Avg Wear Cost    : {baseline.avg_wear_cost:.4f}")
    print(f"  Throughput       : {baseline.throughput_ops_per_s:.2f} ops/s")
    print()

    print(f"- {optimized.name}")
    print(f"  Avg Latency      : {optimized.avg_latency_ms:.4f} ms")
    print(f"  Avg Energy       : {optimized.avg_energy_mj:.4f} mJ/op")
    print(f"  Avg Wear Cost    : {optimized.avg_wear_cost:.4f}")
    print(f"  Throughput       : {optimized.throughput_ops_per_s:.2f} ops/s")
    print()

    print("Relative Improvement (AI vs Baseline)")
    print(f"- Latency reduction        : {latency_gain:.2f}%")
    print(f"- Energy reduction         : {energy_gain:.2f}%")
    print(f"- Wear-cost reduction      : {wear_gain:.2f}%")
    print(f"- Throughput increase      : {throughput_gain:.2f}%")

    print()
    print("Interpretation")
    print("- Energy-aware placement can improve edge AI responsiveness.")
    print("- Keeping hot/reused blocks in faster flash zones lowers access overhead.")
    print("- Capacity-aware scheduling avoids overloading premium zones.")


def run_simulation(
    count: int = 220,
    seed: int = 123,
    workloads: list[WorkloadProfile] | None = None,
) -> tuple[PlacementResult, PlacementResult, int]:
    """Run the baseline and AI-inspired strategies on a workload collection."""
    flash = EdgeFlashModel()
    workloads = workloads if workloads is not None else generate_workloads(count=count, seed=seed)
    baseline = simulate_baseline(workloads, flash)
    optimized = simulate_ai_optimized(workloads, flash)
    return baseline, optimized, len(workloads)


def main() -> None:
    baseline, optimized, total_blocks = run_simulation(count=220, seed=123)
    print_report(baseline, optimized, total_blocks=total_blocks)


if __name__ == "__main__":
    main()
