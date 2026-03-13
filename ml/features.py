from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "access_frequency",
    "write_ratio",
    "temporal_reuse",
    "block_size_kb",
    "size_factor",
    "read_bias",
]


def profile_to_feature_row(profile) -> dict[str, float]:
    return {
        "access_frequency": float(profile.access_frequency),
        "write_ratio": float(profile.write_ratio),
        "temporal_reuse": float(profile.temporal_reuse),
        "block_size_kb": float(profile.block_size_kb),
        "size_factor": float(profile.block_size_kb) / 16.0,
        "read_bias": 1.0 - float(profile.write_ratio),
    }


def workloads_to_frame(workloads) -> pd.DataFrame:
    rows = [profile_to_feature_row(profile) for profile in workloads]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def workloads_with_labels_to_frame(workloads, labels) -> pd.DataFrame:
    frame = workloads_to_frame(workloads)
    frame["label"] = list(labels)
    return frame