from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from ml.features import FEATURE_COLUMNS, profile_to_feature_row


@dataclass
class ZonePrediction:
    zone: str
    priority_score: float
    probabilities: dict[str, float]


def model_path():
    from ml.training import MODEL_PATH

    return MODEL_PATH


def load_trained_model():
    current_model_path = model_path()
    if not current_model_path.exists():
        return None
    return joblib.load(current_model_path)


def get_model_evaluation(artifact) -> dict[str, object] | None:
    if artifact is None:
        return None
    return artifact.get("evaluation")


def predict_profile_zone(profile, artifact) -> ZonePrediction:
    if artifact is None:
        hotness = 0.50 * profile.access_frequency + 0.30 * profile.temporal_reuse + 0.20 * (1.0 - profile.write_ratio)
        zone = "HOT_CACHE" if hotness >= 0.70 else "BALANCED" if hotness >= 0.38 else "COLD_DENSE"
        return ZonePrediction(zone=zone, priority_score=hotness, probabilities={zone: 1.0})

    model = artifact["model"]
    row = profile_to_feature_row(profile)
    feature_frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    predicted_zone = model.predict(feature_frame)[0]
    probability_values = model.predict_proba(feature_frame)[0]
    probabilities = dict(zip(model.classes_, probability_values))
    priority_score = float(probabilities.get("HOT_CACHE", 0.0) + 0.5 * probabilities.get("BALANCED", 0.0))
    return ZonePrediction(zone=predicted_zone, priority_score=priority_score, probabilities=probabilities)