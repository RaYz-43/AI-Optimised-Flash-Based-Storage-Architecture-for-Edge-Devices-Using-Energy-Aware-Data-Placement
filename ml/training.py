from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from edge_ai_flash_project import generate_workloads
from ml.features import FEATURE_COLUMNS, workloads_to_frame, workloads_with_labels_to_frame


MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.joblib"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "training_workloads.csv"
ZONE_NAMES = ["HOT_CACHE", "BALANCED", "COLD_DENSE"]


@dataclass
class TrainingArtifacts:
    model_path: Path
    dataset_path: Path
    sample_count: int
    evaluation: dict[str, object]


def heuristic_hotness(profile) -> float:
    return 0.50 * profile.access_frequency + 0.30 * profile.temporal_reuse + 0.20 * (1.0 - profile.write_ratio)


def preferred_zone_for_profile(profile) -> str:
    hotness = heuristic_hotness(profile)

    if hotness >= 0.64 and profile.write_ratio <= 0.64 and profile.block_size_kb <= 32:
        return "HOT_CACHE"
    if hotness <= 0.34 or (hotness <= 0.40 and profile.block_size_kb >= 32):
        return "COLD_DENSE"
    return "BALANCED"


def build_classifier(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
    )


def build_candidates(seed: int):
    """Candidate models for benchmark comparison on identical holdout data."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": build_classifier(seed),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=seed,
        ),
    }


def generate_labeled_workloads(sample_count: int = 1200, seed: int = 2026):
    workloads = generate_workloads(count=sample_count, seed=seed)
    labels = [preferred_zone_for_profile(profile) for profile in workloads]
    return workloads, labels


def evaluate_classifier(classifier, train_features, test_features, train_labels, test_labels) -> dict[str, object]:
    classifier.fit(train_features[FEATURE_COLUMNS], train_labels)
    predicted_labels = classifier.predict(test_features[FEATURE_COLUMNS])

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        test_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        test_labels,
        predicted_labels,
        labels=ZONE_NAMES,
        average=None,
        zero_division=0,
    )
    confusion = confusion_matrix(test_labels, predicted_labels, labels=ZONE_NAMES)

    return {
        "train_size": int(len(train_features)),
        "test_size": int(len(test_features)),
        "accuracy": float(accuracy_score(test_labels, predicted_labels)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label: {
                "precision": float(class_precision[index]),
                "recall": float(class_recall[index]),
                "f1": float(class_f1[index]),
                "support": int(class_support[index]),
            }
            for index, label in enumerate(ZONE_NAMES)
        },
        "confusion_matrix": {
            "labels": ZONE_NAMES,
            "values": confusion.tolist(),
        },
    }


def benchmark_models(features, labels, seed: int) -> tuple[dict[str, dict[str, object]], str]:
    label_counts = Counter(labels)
    use_stratify = all(label_counts.get(label, 0) >= 2 for label in ZONE_NAMES)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels if use_stratify else None,
    )

    comparison: dict[str, dict[str, object]] = {}
    best_model_name = "Random Forest"
    best_macro_f1 = -1.0

    for model_name, model in build_candidates(seed).items():
        evaluation = evaluate_classifier(model, train_features, test_features, train_labels, test_labels)
        comparison[model_name] = evaluation
        if float(evaluation["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(evaluation["macro_f1"])
            best_model_name = model_name

    return comparison, best_model_name


def train_and_save_model(sample_count: int = 1200, seed: int = 2026) -> TrainingArtifacts:
    workloads, labels = generate_labeled_workloads(sample_count=sample_count, seed=seed)
    feature_frame = workloads_to_frame(workloads)
    comparison, best_model_name = benchmark_models(feature_frame, labels, seed=seed)
    evaluation = comparison[best_model_name]

    classifier = build_candidates(seed)[best_model_name]
    classifier.fit(feature_frame[FEATURE_COLUMNS], labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_frame = workloads_with_labels_to_frame(workloads, labels)
    dataset_frame.to_csv(DATASET_PATH, index=False)

    joblib.dump(
        {
            "model": classifier,
            "model_name": best_model_name,
            "feature_columns": FEATURE_COLUMNS,
            "labels": ZONE_NAMES,
            "seed": seed,
            "sample_count": sample_count,
            "evaluation": evaluation,
            "comparison": comparison,
        },
        MODEL_PATH,
    )

    return TrainingArtifacts(model_path=MODEL_PATH, dataset_path=DATASET_PATH, sample_count=sample_count, evaluation=evaluation)


if __name__ == "__main__":
    artifacts = train_and_save_model()
    artifact = joblib.load(artifacts.model_path)
    comparison = artifact.get("comparison", {})
    model_name = artifact.get("model_name", "Random Forest")

    print("model comparison (same holdout split)")
    print(f"{'model':<24} {'accuracy':>10} {'macro_f1':>10} {'weighted_f1':>12}")
    for name, metrics in comparison.items():
        marker = "  <- selected" if name == model_name else ""
        print(
            f"{name:<24} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['weighted_f1']:>12.4f}{marker}"
        )

    print(f"saved model to {artifacts.model_path}")
    print(f"saved training data to {artifacts.dataset_path}")
    print(f"selected model: {model_name}")
    print(f"holdout accuracy: {artifacts.evaluation['accuracy']:.4f}")
    print(f"holdout macro f1: {artifacts.evaluation['macro_f1']:.4f}")
    print(f"holdout weighted f1: {artifacts.evaluation['weighted_f1']:.4f}")