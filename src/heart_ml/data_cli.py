"""Command-line entry point for dataset download."""
import argparse

from heart_ml.data import download_dataset

parser = argparse.ArgumentParser(description="Download the UCI Heart Disease dataset.")
parser.add_argument("--output", default="data/raw/heart_disease.csv")
args = parser.parse_args()
print(f"Saved dataset to: {download_dataset(args.output)}")
