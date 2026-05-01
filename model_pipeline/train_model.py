from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Canonical 8-feature posture design (has_phone removed).
FEATURE_ORDER = [
    "neck_ratio",
    "forward_lean_z",
    "shoulder_tilt_ratio",
    "head_tilt_ratio",
    "hand_to_face_ratio",
    "pose_x",
    "pose_y",
    "wrist_elevated",
]

# Backward-compatible aliases for legacy datasets.
FEATURE_ALIASES = {
    "neck_ratio": ["neck_ratio", "neckneck_ratio"],
    "forward_lean_z": ["forward_lean_z"],
    "shoulder_tilt_ratio": ["shoulder_tilt_ratio", "shoulder_tilt"],
    "head_tilt_ratio": ["head_tilt_ratio", "head_tilt"],
    "hand_to_face_ratio": ["hand_to_face_ratio", "hand_to_face"],
    "pose_x": ["pose_x"],
    "pose_y": ["pose_y"],
    "wrist_elevated": ["wrist_elevated"],
}


def load_dataset() -> pd.DataFrame:
    candidates = [
        Path("/kaggle/input/datasets/minorin2847/posture-dataset/posture_dataset.csv"),
        Path("data/posture_dataset.csv"),
    ]
    for path in candidates:
        if path.exists():
            print(f"Loading dataset: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError("Could not find posture_dataset.csv in expected locations.")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    selected = {}
    for canonical_name in FEATURE_ORDER:
        source = next((name for name in FEATURE_ALIASES[canonical_name] if name in data.columns), None)
        if source is None:
            raise ValueError(
                f"Missing required feature for '{canonical_name}'. "
                f"Expected one of: {FEATURE_ALIASES[canonical_name]}"
            )
        selected[canonical_name] = data[source]

    feature_df = pd.DataFrame(selected)
    return feature_df


def build_models() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=15,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    df = load_dataset().dropna().copy()
    print(f"Total valid samples: {len(df)}")

    X = build_feature_frame(df)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }

    models = build_models()
    cv_results = {}

    print("\n=== Cross-validation (leakage-safe pipelines) ===")
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        summary = {metric: float(np.mean(values)) for metric, values in scores.items() if metric.startswith("test_")}
        cv_results[name] = summary
        print(
            f"{name:14s} | "
            f"acc={summary['test_accuracy']:.4f} | "
            f"f1_macro={summary['test_f1_macro']:.4f} | "
            f"bal_acc={summary['test_balanced_accuracy']:.4f}"
        )

    best_name = max(
        cv_results,
        key=lambda model_name: (
            cv_results[model_name]["test_f1_macro"],
            cv_results[model_name]["test_accuracy"],
        ),
    )
    best_pipeline = models[best_name]
    print(f"\nSelected model by CV: {best_name}")

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\n=== Test set evaluation (single final hold-out) ===")
    print(f"accuracy:          {test_acc:.4f}")
    print(f"f1_macro:          {test_f1:.4f}")
    print(f"balanced_accuracy: {test_bal_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    scaler = best_pipeline.named_steps["scaler"]
    model = best_pipeline.named_steps["model"]
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(model, models_dir / "best_posture_model.pkl")

    print("\nSaved artifacts:")
    print(f"- {models_dir / 'scaler.pkl'}")
    print(f"- {models_dir / 'best_posture_model.pkl'}")


if __name__ == "__main__":
    main()

