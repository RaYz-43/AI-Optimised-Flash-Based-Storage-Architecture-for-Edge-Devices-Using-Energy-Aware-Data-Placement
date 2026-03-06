"""
Project 2: AI-Optimised Flash-Based Storage Architecture for Edge Devices
Single-file prototype for learning and demonstration.

How to run:
    python edge_ai_flash_project.py
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
import random


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
        self.zones = {
            "HOT_CACHE": {"latency_ms": 0.45, "energy_mj": 0.55, "wear_factor": 1.05, "capacity_blocks": 50},
            "BALANCED": {"latency_ms": 0.80, "energy_mj": 0.85, "wear_factor": 1.00, "capacity_blocks": 80},
            "COLD_DENSE": {"latency_ms": 1.30, "energy_mj": 1.10, "wear_factor": 0.90, "capacity_blocks": 120},
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


def generate_workloads(count: int, seed: int = 42) -> list[WorkloadProfile]:
    random.seed(seed)
    workloads: list[WorkloadProfile] = []

    for i in range(count):
        # Create mixed AI-edge workloads: telemetry, video snippets, model cache, etc.
        profile = WorkloadProfile(
            block_id=i,
            access_frequency=min(1.0, max(0.0, random.gauss(0.45, 0.25))),
            write_ratio=min(1.0, max(0.0, random.gauss(0.40, 0.20))),
            temporal_reuse=min(1.0, max(0.0, random.gauss(0.50, 0.30))),
            block_size_kb=random.choice([4, 8, 16, 32, 64]),
        )
        workloads.append(profile)

    return workloads


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

        # Access locality bonus for hot and highly reused data in fast zone.
        if zone_name == "HOT_CACHE" and (w.access_frequency > 0.6 and w.temporal_reuse > 0.55):
            latency *= 0.85
            energy *= 0.90

        # Write-heavy data in cold zone gets a latency penalty.
        if zone_name == "COLD_DENSE" and w.write_ratio > 0.65:
            latency *= 1.12

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


def main() -> None:
    flash = EdgeFlashModel()
    workloads = generate_workloads(count=220, seed=123)

    baseline = simulate_baseline(workloads, flash)
    optimized = simulate_ai_optimized(workloads, flash)

    print_report(baseline, optimized, total_blocks=len(workloads))


if __name__ == "__main__":
    main()
