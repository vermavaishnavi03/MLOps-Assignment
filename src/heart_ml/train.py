"""Train, cross-validate, compare and log classification models with MLflow."""
import json
#import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, roc_auc_score, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_ml.data import (
    CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, load_dataset,
)

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])


def model_candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def train(data_path="data/raw/heart_disease.csv", artifact_dir="artifacts"):
    frame = load_dataset(data_path)
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    if y.nunique() < 2:
        raise ValueError("Training requires examples from both target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "roc_auc": "roc_auc"}
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    #mlflow.set_experiment("uci-heart-disease-classification")

    # Use SQLite for MLflow experiment tracking
    # Use a local SQLite database inside the project workspace or artifacts folder
    mlflow_db = Path("artifacts") / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("uci-heart-disease-classification")

    results = []
    fitted = {}
    for name, estimator in model_candidates().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=name) as run:
            cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring,
                                       return_train_score=False, n_jobs=1)
            cv_metrics = {f"cv_{metric}_mean": float(np.mean(cv_result[f"test_{metric}"]))
                          for metric in scoring}
            cv_metrics.update({f"cv_{metric}_std": float(np.std(cv_result[f"test_{metric}"]))
                               for metric in scoring})
            pipeline.fit(X_train, y_train)
            predicted = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = {
                **cv_metrics,
                "test_accuracy": float(accuracy_score(y_test, predicted)),
                "test_precision": float(precision_score(y_test, predicted, zero_division=0)),
                "test_recall": float(recall_score(y_test, predicted, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics(metrics)

            report_path = artifact_path / f"{name}_classification_report.txt"
            report_path.write_text(classification_report(y_test, predicted, zero_division=0),
                                   encoding="utf-8")
            cm_path = artifact_path / f"{name}_confusion_matrix.png"
            plt.figure(figsize=(5, 4))
            from sklearn.metrics import ConfusionMatrixDisplay
            ConfusionMatrixDisplay.from_predictions(y_test, predicted)
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close("all")
            roc_path = artifact_path / f"{name}_roc_curve.png"
            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close("all")
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
            ],)
            results.append({"model": name, "run_id": run.info.run_id, **metrics})
            fitted[name] = pipeline

    results = sorted(results, key=lambda item: item["cv_roc_auc_mean"], reverse=True)
    best = results[0]
    best_model = fitted[best["model"]]
    model_path = artifact_path / "model.joblib"
    joblib.dump(best_model, model_path)
    results_path = artifact_path / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    metadata = {
        "model_name": best["model"],
        "target": "binary heart disease indicator (0=no disease, 1=disease)",
        "features": FEATURE_COLUMNS,
        "selection_metric": "mean 5-fold cross-validation ROC-AUC",
        "best_cv_roc_auc": best["cv_roc_auc_mean"],
        "random_state": RANDOM_STATE,
        "medical_disclaimer": "Educational demonstration only; not for diagnosis or treatment.",
    }
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Selected model: {best['model']} (CV ROC-AUC={best['cv_roc_auc_mean']:.3f})")
    print(f"Saved reusable pipeline to {model_path.resolve()}")
    return results


if __name__ == "__main__":
    data_file = Path("data/raw/heart_disease.csv")
    if not data_file.exists():
        from heart_ml.data import download_dataset
        download_dataset(data_file)
    train(data_file)
