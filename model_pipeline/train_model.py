from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.feature_schema import BASE_FEATURE_ORDER, FEATURE_ALIASES, FEATURE_DEFAULTS, FEATURE_ORDER, NO_HAND_VISIBLE_RATIO

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
LEGACY_GROUP_CHUNK_SIZE = 120


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
            if canonical_name in BASE_FEATURE_ORDER:
                raise ValueError(
                    f"Missing required feature for '{canonical_name}'. "
                    f"Expected one of: {FEATURE_ALIASES[canonical_name]}"
                )
            selected[canonical_name] = FEATURE_DEFAULTS[canonical_name]
            continue
        selected[canonical_name] = pd.to_numeric(data[source], errors="coerce").fillna(
            FEATURE_DEFAULTS.get(canonical_name, np.nan)
        )

    feature_df = pd.DataFrame(selected)
    feature_df["wrist_elevated"] = feature_df["wrist_elevated"].astype(float)
    feature_df["hand_to_face_ratio"] = feature_df["hand_to_face_ratio"].clip(
        lower=0.0,
        upper=NO_HAND_VISIBLE_RATIO,
    )
    return feature_df


def build_sample_groups(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    if "capture_group" in df.columns:
        groups = df["capture_group"].astype("string")
        if groups.notna().any() and (groups.dropna().str.len() > 0).any():
            fallback = build_legacy_groups(y)
            return groups.where(groups.notna() & (groups.str.len() > 0), fallback)

    print(
        "No capture_group column found; using row-order chunks as a leakage-reduction fallback. "
        "Collect new data with data_collection.py for stronger validation."
    )
    return build_legacy_groups(y)


def build_legacy_groups(y: pd.Series) -> pd.Series:
    label_change_id = y.ne(y.shift()).cumsum()
    position_in_run = y.groupby(label_change_id).cumcount()
    chunk_id = position_in_run // LEGACY_GROUP_CHUNK_SIZE
    return y.astype(str) + "_run" + label_change_id.astype(str) + "_chunk" + chunk_id.astype(str)


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = df.copy()
    if "label" not in data.columns:
        raise ValueError("Dataset must include a 'label' column.")

    X = build_feature_frame(data)
    y = data["label"].astype(str).str.strip()
    groups = build_sample_groups(data, y)

    valid_mask = y.ne("") & X.notna().all(axis=1)
    dropped = int((~valid_mask).sum())
    if dropped:
        print(f"Dropped invalid rows: {dropped}")

    return (
        X.loc[valid_mask].reset_index(drop=True),
        y.loc[valid_mask].reset_index(drop=True),
        groups.loc[valid_mask].reset_index(drop=True),
    )


def describe_dataset(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
    print(f"Total valid samples: {len(X)}")
    print(f"Unique groups: {groups.nunique()}")
    print("\nLabel distribution:")
    print(y.value_counts().to_string())

    no_wrist_count = int((X["hand_to_face_ratio"] >= NO_HAND_VISIBLE_RATIO).sum())
    no_wrist_pct = 100.0 * no_wrist_count / len(X)
    print(
        f"\nNo visible wrist / capped hand_to_face samples: "
        f"{no_wrist_count} ({no_wrist_pct:.1f}%)"
    )
    legacy_rows = int((X["shoulder_width_ratio"] == 0.0).sum())
    if legacy_rows:
        legacy_pct = 100.0 * legacy_rows / len(X)
        print(f"Rows using default extended features: {legacy_rows} ({legacy_pct:.1f}%)")


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    if groups.nunique() < 2:
        print("Only one group available; falling back to stratified row split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test, groups.loc[X_train.index], groups.loc[X_test.index]

    splitter = GroupShuffleSplit(n_splits=100, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    all_labels = set(y.unique())
    best_split = None
    best_score = float("inf")
    full_dist = y.value_counts(normalize=True).sort_index()

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_labels = set(y.iloc[train_idx].unique())
        test_labels = set(y.iloc[test_idx].unique())
        if train_labels != all_labels or test_labels != all_labels:
            continue

        test_dist = y.iloc[test_idx].value_counts(normalize=True).reindex(full_dist.index, fill_value=0.0)
        balance_error = float((test_dist - full_dist).abs().sum())
        size_error = abs((len(test_idx) / len(X)) - TEST_SIZE)
        score = balance_error + size_error
        if score < best_score:
            best_score = score
            best_split = (train_idx, test_idx)

    if best_split is None:
        print("Could not create a balanced group split; falling back to stratified row split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test, groups.loc[X_train.index], groups.loc[X_test.index]

    train_idx, test_idx = best_split
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        groups.iloc[train_idx].reset_index(drop=True),
        groups.iloc[test_idx].reset_index(drop=True),
    )


def build_cv(y_train: pd.Series, train_groups: pd.Series):
    min_class_count = int(y_train.value_counts().min())
    groups_per_class = (
        pd.DataFrame({"label": y_train, "group": train_groups})
        .drop_duplicates()
        .groupby("label")["group"]
        .nunique()
    )
    min_group_count = int(groups_per_class.min())
    n_splits = min(CV_SPLITS, min_class_count, min_group_count)

    if train_groups.nunique() >= n_splits and n_splits >= 2:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), train_groups

    n_splits = min(CV_SPLITS, min_class_count)
    if n_splits >= 2:
        print("Not enough groups for grouped CV; using stratified row CV.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None

    raise ValueError("Not enough samples per class for cross-validation.")


def build_models() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_split=20,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        class_weight="balanced",
                        bootstrap=True,
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
                        C=0.5,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    df = load_dataset().copy()

    X, y, groups = prepare_dataset(df)
    describe_dataset(X, y, groups)

    X_train, X_test, y_train, y_test, train_groups, test_groups = group_train_test_split(
        X,
        y,
        groups,
    )
    print(f"\nTraining samples: {len(X_train)} ({train_groups.nunique()} groups)")
    print(f"Testing samples: {len(X_test)} ({test_groups.nunique()} groups)")

    cv, cv_groups = build_cv(y_train, train_groups)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }

    models = build_models()
    cv_results = {}

    print("\n=== Cross-validation (group-aware pipelines) ===")
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
        validation_gap = train_f1 - summary["test_f1_macro"]
        cv_results[name] = summary
        print(
            f"{name:14s} | "
            f"acc={summary['test_accuracy']:.4f} | "
            f"f1_macro={summary['test_f1_macro']:.4f} | "
            f"bal_acc={summary['test_balanced_accuracy']:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"gap={validation_gap:.4f}"
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
    y_train_pred = best_pipeline.predict(X_train)
    y_pred = best_pipeline.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\n=== Test set evaluation (single final hold-out) ===")
    print(f"train_f1_macro:    {train_f1:.4f}")
    print(f"accuracy:          {test_acc:.4f}")
    print(f"f1_macro:          {test_f1:.4f}")
    print(f"balanced_accuracy: {test_bal_acc:.4f}")
    print(f"train/test f1 gap: {train_f1 - test_f1:.4f}")
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

