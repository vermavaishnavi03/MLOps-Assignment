"""Generate reproducible exploratory data analysis plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
#import pandas as pd
import seaborn as sns

from heart_ml.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset


def run_eda(data_path="data/raw/heart_disease.csv", output_dir="reports/figures"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path)
    sns.set_theme(style="whitegrid")

    frame[FEATURE_COLUMNS].hist(figsize=(14, 10), bins=18)
    plt.tight_layout()
    plt.savefig(output / "feature_histograms.png", dpi=160)
    plt.close("all")

    plt.figure(figsize=(12, 9))
    sns.heatmap(frame[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True),
                cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(output / "correlation_heatmap.png", dpi=160)
    plt.close("all")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=frame, x=TARGET_COLUMN)
    plt.title("Class balance (0 = no disease, 1 = disease)")
    plt.tight_layout()
    plt.savefig(output / "class_balance.png", dpi=160)
    plt.close("all")

    summary = frame.describe(include="all").transpose()
    summary.to_csv(output / "descriptive_statistics.csv")
    frame.isna().sum().rename("missing_values").to_csv(output / "missing_values.csv")
    print(f"EDA outputs saved in {output.resolve()}")


if __name__ == "__main__":
    run_eda()
