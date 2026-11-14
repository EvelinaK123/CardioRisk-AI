"""Predict the dataset HeartDisease label from unlabeled CSV rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from eda import FEATURE_COLUMNS
from train import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with the 11 predictor columns; HeartDisease is not required.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the results CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Saved preprocessing-and-model pipeline.")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}")
    if not args.model.is_file():
        parser.error(
            f"Saved model not found: {args.model}. Run src/train.py first in this Codespace."
        )

    input_frame = pd.read_csv(args.input)
    missing = [column for column in FEATURE_COLUMNS if column not in input_frame.columns]
    if missing:
        parser.error(f"Prediction CSV is missing required predictors: {', '.join(missing)}")

    # HeartDisease is never needed for prediction. Any extra columns are ignored.
    features = input_frame[FEATURE_COLUMNS]
    pipeline = joblib.load(args.model)
    predicted = pipeline.predict(features)

    results = input_frame.copy()
    results["predicted_HeartDisease"] = predicted.astype(int)

    if hasattr(pipeline, "predict_proba"):
        classifier = pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        if 1 in classes:
            score = pipeline.predict_proba(features)[:, classes.index(1)]
            results["score_for_label_1"] = score
    elif hasattr(pipeline, "decision_function"):
        results["score_for_label_1"] = pipeline.decision_function(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Predictions written to: {args.output}")
    print("predicted_HeartDisease is a predicted dataset class, not a diagnosis.")
    if "score_for_label_1" in results.columns:
        print("score_for_label_1 is a model score for dataset label 1, not a clinical probability.")


if __name__ == "__main__":
    main()
