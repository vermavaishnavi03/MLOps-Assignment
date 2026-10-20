import pandas as pd

from heart_ml.data import FEATURE_COLUMNS
from heart_ml.train import build_preprocessor, model_candidates


def test_preprocessor_handles_missing_values_and_unseen_categories():
    rows = [
        [60, 1, 2, 130, 240, 0, 1, 150, 0, 1.0, 1, 0, 2],
        [45, 0, 1, 120, 220, 0, 0, 170, 0, 0.0, 2, 0, 3],
        [55, 1, 3, 140, 260, 1, 2, 140, 1, 2.0, 1, 1, 2],
        [50, 0, 0, 125, 210, 0, 1, 165, 0, 0.5, 2, 0, 3],
    ]
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    frame.loc[0, "chol"] = float("nan")
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(frame)
    assert transformed.shape[0] == 4


def test_two_classification_models_are_provided():
    models = model_candidates()
    assert "logistic_regression" in models
    assert "random_forest" in models
