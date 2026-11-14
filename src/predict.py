"""Predict the dataset HeartDisease label from unlabeled CSV rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from eda import FEATURE_COLUMNS
from train import DEFAULT_MODEL_PATH, HIGH_SCORE_MIN, LOW_SCORE_MAX


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
            results["risk_band"] = [
                "Low" if value < LOW_SCORE_MAX
                else "Medium" if value < HIGH_SCORE_MIN
                else "High"
                for value in score
            ]
    elif hasattr(pipeline, "decision_function"):
        results["score_for_label_1"] = pipeline.decision_function(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Predictions written to: {args.output}")
    print("predicted_HeartDisease is the predicted binary HeartDisease label from this dataset.")
    if "score_for_label_1" in results.columns:
        print("score_for_label_1 is the model's score for dataset label 1.")
    if "risk_band" in results.columns:
        print("risk_band uses Low < 0.33, Medium >= 0.33 and < 0.67, High >= 0.67.")
        print("These are project score bands, not clinically validated risk thresholds.")
    elif "score_for_label_1" in results.columns:
        print("No risk_band was created because the loaded model did not return class probabilities.")


if __name__ == "__main__":
    main()
