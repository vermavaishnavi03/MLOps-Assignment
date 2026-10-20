"""Data acquisition and validation helpers for the UCI Heart Disease dataset."""
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def download_dataset(output_path: str | Path = "data/raw/heart_disease.csv") -> Path:
    """Download UCI Heart Disease data via the official UCI repository package."""
    from ucimlrepo import fetch_ucirepo

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    # Use canonical field names regardless of the names supplied by the package.
    features.columns = [str(c).strip().lower() for c in features.columns]
    targets.columns = [str(c).strip().lower() for c in targets.columns]
    # The UCI Cleveland subset uses "num" as the disease-severity target.
    target_name = "num" if "num" in targets.columns else targets.columns[0]
    frame = features.copy()
    frame[TARGET_COLUMN] = pd.to_numeric(targets[target_name], errors="coerce").to_numpy()
    frame = frame.rename(columns={"num": TARGET_COLUMN})
    # UCI target 0 = no disease; 1–4 = disease present.
    frame[TARGET_COLUMN] = (pd.to_numeric(frame[TARGET_COLUMN], errors="coerce") > 0).astype("int64")
    # Retain the expected 13 input columns in a stable order.
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"UCI data schema differs from the expected Cleveland schema. Missing: {missing}. "
            f"Received columns: {list(frame.columns)}"
        )
    frame = frame[FEATURE_COLUMNS + [TARGET_COLUMN]]
    frame.to_csv(output, index=False)
    return output


def load_dataset(path: str | Path = "data/raw/heart_disease.csv") -> pd.DataFrame:
    """Load and validate a locally stored dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m heart_ml.data_cli` to download it."
        )
    frame = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame[required].copy()
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TARGET_COLUMN])
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary: 0 (absence) or 1 (presence).")
    return frame
