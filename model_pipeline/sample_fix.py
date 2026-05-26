from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.feature_schema import (
    BASE_FEATURE_ORDER,
    FEATURE_ALIASES,
    FEATURE_DEFAULTS,
    FEATURE_ORDER,
    METADATA_COLUMNS,
    NO_HAND_VISIBLE_RATIO,
    OUTPUT_COLUMNS,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    ROOT_DIR / "data" / "posture_dataset_train.csv",
    ROOT_DIR / "data" / "posture_dataset_test.csv",
]

LABELS_TO_REMOVE = {"Absence", "Using Phone"}
DEFAULT_NEAR_DUPLICATE_RADIUS = 0.03


def read_csv_lenient(path: Path) -> pd.DataFrame:
    cleaned_rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path} is empty.")

        # Legacy exports sometimes had an extra hand_to_ear column.
        if len(header) == 11 and "hand_to_ear" in header:
            header.pop(header.index("hand_to_ear"))
        elif len(header) == 11:
            header.pop(5)

        for row in reader:
            if not row:
                continue
            if len(row) == 11 and len(header) == 10:
                row.pop(5)
            if len(row) == len(header):
                cleaned_rows.append(row)

    return pd.DataFrame(cleaned_rows, columns=header)


def first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    return next((name for name in aliases if name in df.columns), None)


def canonicalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    selected = {}

    for column in METADATA_COLUMNS:
        selected[column] = df[column] if column in df.columns else ""

    for feature_name, aliases in FEATURE_ALIASES.items():
        source = first_existing_column(df, aliases)
        if source is None:
            if feature_name in BASE_FEATURE_ORDER:
                raise ValueError(f"Missing required feature '{feature_name}'. Expected one of: {aliases}")
            selected[feature_name] = FEATURE_DEFAULTS[feature_name]
            continue
        selected[feature_name] = pd.to_numeric(df[source], errors="coerce").fillna(
            FEATURE_DEFAULTS.get(feature_name, np.nan)
        )

    if "label" not in df.columns:
        raise ValueError("Missing required 'label' column.")
    selected["label"] = df["label"].astype(str).str.strip()

    canonical = pd.DataFrame(selected, columns=OUTPUT_COLUMNS)
    canonical["wrist_elevated"] = canonical["wrist_elevated"].astype(float).round().clip(0, 1)
    for binary_column in ["face_detected", "hand_visible"]:
        canonical[binary_column] = canonical[binary_column].astype(float).round().clip(0, 1)
    canonical["visible_wrist_count"] = canonical["visible_wrist_count"].astype(float).round().clip(0, 2)
    canonical["hand_to_face_ratio"] = canonical["hand_to_face_ratio"].clip(
        lower=0.0,
        upper=NO_HAND_VISIBLE_RATIO,
    )
    return canonical


def remove_near_duplicates(df: pd.DataFrame, radius: float) -> pd.DataFrame:
    if radius <= 0 or df.empty:
        return df

    keep_indices = []
    feature_df = df[FEATURE_ORDER].astype(float)

    for _, label_indices in df.groupby("label", sort=False).groups.items():
        indices = list(label_indices)
        label_features = feature_df.loc[indices]
        if len(label_features) <= 1:
            keep_indices.extend(indices)
            continue

        scaled = StandardScaler().fit_transform(label_features)
        neighbors = NearestNeighbors(radius=radius).fit(scaled)
        neighbor_graph = neighbors.radius_neighbors_graph(scaled, mode="connectivity")

        excluded = set()
        for local_index in range(neighbor_graph.shape[0]):
            if local_index in excluded:
                continue
            keep_indices.append(indices[local_index])
            near_indices = neighbor_graph[local_index].nonzero()[1]
            excluded.update(int(i) for i in near_indices if int(i) != local_index)

    keep_indices.sort()
    return df.loc[keep_indices].reset_index(drop=True)


def clean_dataset(
    path: Path,
    near_duplicate_radius: float,
    shuffle: bool,
    dry_run: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw_df = read_csv_lenient(path)
    original_count = len(raw_df)
    df = canonicalize_dataset(raw_df)

    df = df.dropna(subset=FEATURE_ORDER + ["label"])
    df = df[df["label"].ne("")]
    loaded_count = len(df)

    df = df[~df["label"].isin(LABELS_TO_REMOVE)]
    label_filtered_count = len(df)

    df = df.drop_duplicates(subset=FEATURE_ORDER + ["label"]).reset_index(drop=True)
    exact_duplicate_count = label_filtered_count - len(df)

    before_near_duplicate_count = len(df)
    df = remove_near_duplicates(df, near_duplicate_radius)
    near_duplicate_count = before_near_duplicate_count - len(df)

    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    final_count = len(df)

    print(f"\n=== {path} ===")
    print(f"Raw rows:                         {original_count}")
    print(f"Dropped invalid rows:             {original_count - loaded_count}")
    print(f"Dropped Absence/Using Phone rows: {loaded_count - label_filtered_count}")
    print(f"Dropped exact duplicates:         {exact_duplicate_count}")
    print(f"Dropped near duplicates:          {near_duplicate_count}")
    print(f"Final rows:                       {final_count}")
    print("Labels:")
    print(df["label"].value_counts().to_string())

    if not dry_run:
        df.to_csv(path, index=False)
        print(f"Saved cleaned dataset: {path}")
    else:
        print("Dry run only; file was not modified.")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize posture CSV columns, cap hand_to_face sentinel values, "
            "remove old labels, and remove exact/near-duplicate samples."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_DATASETS,
        help="CSV files to clean. Defaults to data/posture_dataset_train.csv and data/posture_dataset_test.csv.",
    )
    parser.add_argument(
        "--near-duplicate-radius",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_RADIUS,
        help=(
            "Radius in standardized feature space for near-duplicate removal. "
            "Use 0 to disable near-duplicate filtering."
        ),
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows after cleaning. Disabled by default.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()

    for path in args.paths:
        clean_dataset(
            path=path,
            near_duplicate_radius=args.near_duplicate_radius,
            shuffle=args.shuffle,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
