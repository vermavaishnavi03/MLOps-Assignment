import pandas as pd
import pytest

from heart_ml.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset


def test_load_dataset_validates_and_keeps_required_columns(tmp_path):
    row = {column: 1 for column in FEATURE_COLUMNS}
    row[TARGET_COLUMN] = 0
    path = tmp_path / "sample.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    result = load_dataset(path)
    assert list(result.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert result[TARGET_COLUMN].iloc[0] == 0


def test_load_dataset_rejects_non_binary_target(tmp_path):
    row = {column: 1 for column in FEATURE_COLUMNS}
    row[TARGET_COLUMN] = 3
    path = tmp_path / "sample.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_dataset(path)


def test_load_dataset_missing_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")
