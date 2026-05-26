from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate

from train_model import (
    RANDOM_STATE,
    build_cv,
    build_models,
    describe_dataset,
    prepare_dataset,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "posture_dataset_train.csv"
DEFAULT_TEST_PATH = ROOT_DIR / "data" / "posture_dataset_test.csv"


def load_named_dataset(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")

    print(f"Loading {name} dataset: {path}")
    return pd.read_csv(path)


def warn_label_mismatch(y_train: pd.Series, y_test: pd.Series) -> None:
    train_labels = set(y_train.unique())
    test_labels = set(y_test.unique())
    missing_from_train = sorted(test_labels - train_labels)
    missing_from_test = sorted(train_labels - test_labels)

    if missing_from_train:
        print(f"WARNING: test labels not present in training data: {missing_from_train}")
    if missing_from_test:
        print(f"WARNING: training labels not present in test data: {missing_from_test}")


def choose_model_with_cv(X_train: pd.DataFrame, y_train: pd.Series, train_groups: pd.Series):
    cv, cv_groups = build_cv(y_train, train_groups)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }

    models = build_models()
    cv_results = {}

    print("\n=== Training-data CV model selection ===")
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            groups=cv_groups,
            n_jobs=-1,
            return_train_score=True,
        )
        summary = {metric: float(np.mean(values)) for metric, values in scores.items() if metric.startswith("test_")}
        train_f1 = float(np.mean(scores["train_f1_macro"]))
        gap = train_f1 - summary["test_f1_macro"]
        cv_results[name] = summary

        print(
            f"{name:14s} | "
            f"acc={summary['test_accuracy']:.4f} | "
            f"f1_macro={summary['test_f1_macro']:.4f} | "
            f"bal_acc={summary['test_balanced_accuracy']:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"gap={gap:.4f}"
        )

    best_name = max(
        cv_results,
        key=lambda model_name: (
            cv_results[model_name]["test_f1_macro"],
            cv_results[model_name]["test_accuracy"],
        ),
    )
    return best_name, models[best_name]


def evaluate_external_test(best_name: str, pipeline, X_train, y_train, X_test, y_test) -> None:
    print(f"\nSelected model by training CV: {best_name}")
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    print("\n=== External-person test evaluation ===")
    print(f"train_f1_macro:    {train_f1:.4f}")
    print(f"accuracy:          {test_acc:.4f}")
    print(f"f1_macro:          {test_f1:.4f}")
    print(f"balanced_accuracy: {test_bal_acc:.4f}")
    print(f"train/test f1 gap: {train_f1 - test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))


def save_artifacts(pipeline, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(model, output_dir / "best_posture_model.pkl")

    print("\nSaved artifacts:")
    print(f"- {output_dir / 'scaler.pkl'}")
    print(f"- {output_dir / 'best_posture_model.pkl'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train on posture_dataset_train.csv and evaluate on posture_dataset_test.csv from a different person. "
            "Use sample_fix.py first if either CSV still contains duplicate/legacy rows."
        )
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--save-model", action="store_true", help="Overwrite models/scaler.pkl and models/best_posture_model.pkl.")
    parser.add_argument("--model-dir", type=Path, default=ROOT_DIR / "models")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    train_df = load_named_dataset("training", args.train_csv)
    test_df = load_named_dataset("external test", args.test_csv)

    print("\n=== Training dataset ===")
    X_train, y_train, train_groups = prepare_dataset(train_df)
    describe_dataset(X_train, y_train, train_groups)

    print("\n=== External test dataset ===")
    X_test, y_test, test_groups = prepare_dataset(test_df)
    describe_dataset(X_test, y_test, test_groups)
    warn_label_mismatch(y_train, y_test)

    best_name, best_pipeline = choose_model_with_cv(X_train, y_train, train_groups)
    evaluate_external_test(best_name, best_pipeline, X_train, y_train, X_test, y_test)

    if args.save_model:
        save_artifacts(best_pipeline, args.model_dir)
    else:
        print("\nArtifacts not saved. Pass --save-model to overwrite the production model files.")


if __name__ == "__main__":
    main()
