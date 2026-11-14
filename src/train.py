"""Compare models safely and evaluate the selected pipeline on a held-out test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda import FEATURE_COLUMNS, TARGET_COLUMN, write_eda_report

CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
NUMERIC_COLUMNS = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_COLUMNS]
DEFAULT_MODEL_PATH = Path("models/heart_disease_pipeline.joblib")

# Fixed project score bands for the model's score assigned to dataset label 1.
# These do not represent clinically validated risk thresholds.
LOW_SCORE_MAX = 0.33
HIGH_SCORE_MIN = 0.67


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build transformers that are fitted only when the pipeline is fitted."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def get_class_1_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray | None:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        classes = list(classifier.classes_)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
    if hasattr(pipeline, "decision_function"):
        decision = pipeline.decision_function(features)
        classes = list(classifier.classes_)
        if classes == [0, 1]:
            return decision
    return None


def safe_auc(y_true: pd.Series, scores: np.ndarray | None) -> float | None:
    if scores is None or len(pd.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def make_evaluation_markdown(
    selected_name: str,
    cv_comparison: dict,
    metrics: dict,
    confusion: list[list[int]],
    input_path: Path,
    rows_excluded_missing_target: int,
) -> str:
    comparison_lines = [
        "| Model | CV balanced accuracy (mean ± SD) | CV accuracy (mean ± SD) |",
        "|---|---:|---:|",
    ]
    for name, result in cv_comparison.items():
        comparison_lines.append(
            f"| {name} | {result['balanced_accuracy']['mean']:.3f} ± {result['balanced_accuracy']['std']:.3f} "
            f"| {result['accuracy']['mean']:.3f} ± {result['accuracy']['std']:.3f} |"
        )

    metric_lines = ["| Metric | Test result |", "|---|---:|"]
    for name in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    ]:
        value = metrics[name]
        formatted = "not available" if value is None else f"{value:.3f}"
        metric_lines.append(f"| {name} | {formatted} |")

    tn, fp = confusion[0]
    fn, tp = confusion[1]
    return "\n".join(
        [
            "# Model evaluation report",
            "",
            f"Training input: {input_path}",
            "",
            "## Model comparison using training data only",
            "",
            *comparison_lines,
            "",
            f"Selected pipeline: **{selected_name}**. Selection used mean cross-validation balanced accuracy on the training portion only. The held-out test set was not used to select or tune a model.",
            "",
            "## Final held-out test results",
            "",
            *metric_lines,
            "",
            "## Score bands used by prediction",
            "",
            f"Risk bands are applied to the model's score for dataset label 1: Low (< {LOW_SCORE_MAX:.2f}), Medium (>= {LOW_SCORE_MAX:.2f} and < {HIGH_SCORE_MIN:.2f}), High (>= {HIGH_SCORE_MIN:.2f}). The model is trained on the binary target; these bands are applied afterward, not learned as three classes. The cutoffs are project score bands, not clinically validated risk thresholds. The test metrics above evaluate the binary target using the model's default prediction threshold.",
            "",
            "Confusion matrix with actual labels as rows (0, 1) and predictions as columns (0, 1):",
            "",
            "| Actual / predicted | 0 | 1 |",
            "|---|---:|---:|",
            f"| 0 | {tn} (true negative) | {fp} (false positive) |",
            f"| 1 | {fn} (false negative) | {tp} (true positive) |",
            "",
            f"Rows with a missing HeartDisease target were excluded from supervised training: {rows_excluded_missing_target}. Zero Cholesterol and zero RestingBP values were kept as numeric observations; they were not changed or used to remove rows.",
            "",
            "For this dataset, a false positive is a row whose dataset label is 0 but the model predicts 1. A false negative is a row whose dataset label is 1 but the model predicts 0. These errors are measured against the dataset labels, not clinical outcomes.",
            "",
            "This report describes a held-out evaluation of this dataset. A high score does not establish clinical usefulness or justify medical decisions.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/heart.csv"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.10 and 0.40.")

    if not args.input.is_file():
        parser.error(
            f"CSV not found: {args.input}. Put the private dataset at data/heart.csv "
            "or pass its path with --input."
        )

    frame = pd.read_csv(args.input)
    # Write the inspection report before any model fitting.
    write_eda_report(frame, args.report_dir / "eda_report.md", str(args.input))

    missing_columns = [
        name for name in FEATURE_COLUMNS + [TARGET_COLUMN] if name not in frame.columns
    ]
    if missing_columns:
        parser.error(f"Missing required CSV columns: {', '.join(missing_columns)}")

    labeled = frame.loc[frame[TARGET_COLUMN].notna(), FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    missing_target_rows = int(frame[TARGET_COLUMN].isna().sum())
    invalid_target = ~labeled[TARGET_COLUMN].isin([0, 1])
    if invalid_target.any():
        invalid_values = sorted(labeled.loc[invalid_target, TARGET_COLUMN].astype(str).unique())
        parser.error(f"HeartDisease must contain only 0 and 1; found: {invalid_values}")

    if labeled.empty:
        parser.error("No rows have a HeartDisease target label.")

    y = labeled[TARGET_COLUMN].astype(int)
    X = labeled[FEATURE_COLUMNS]
    if y.nunique() != 2:
        parser.error("Training requires both HeartDisease dataset labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    min_train_class_count = int(y_train.value_counts().min())
    cv_splits = min(5, min_train_class_count)
    if cv_splits < 2:
        parser.error(
            "Not enough training rows in the smaller target class for stratified "
            "cross-validation; add more labeled data or use a larger dataset."
        )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    specificity_scorer = make_scorer(specificity_score)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall_sensitivity": make_scorer(recall_score, zero_division=0),
        "specificity": specificity_scorer,
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=args.random_state),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=args.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    cv_comparison: dict[str, dict[str, dict[str, float]]] = {}
    for name, pipeline in candidates.items():
        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        cv_comparison[name] = {}
        for metric in scoring:
            values = np.asarray(results[f"test_{metric}"], dtype=float)
            cv_comparison[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }

    # The stable insertion order makes Logistic Regression the tie-breaker.
    selected_name = max(
        candidates,
        key=lambda name: cv_comparison[name]["balanced_accuracy"]["mean"],
    )
    selected_pipeline = candidates[selected_name]
    selected_pipeline.fit(X_train, y_train)

    predictions = selected_pipeline.predict(X_test)
    scores = get_class_1_scores(selected_pipeline, X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": safe_auc(y_test, scores),
        "confusion_matrix_labels_0_then_1": matrix.astype(int).tolist(),
        "risk_band_thresholds_for_score_of_label_1": {
            "low": {"max_exclusive": LOW_SCORE_MAX},
            "medium": {"min_inclusive": LOW_SCORE_MAX, "max_exclusive": HIGH_SCORE_MIN},
            "high": {"min_inclusive": HIGH_SCORE_MIN},
        },
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
        "rows_excluded_for_missing_target": missing_target_rows,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, args.model_output)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    comparison_payload = {
        "selection_metric": "mean cross-validation balanced_accuracy on training data",
        "test_set_used_for_selection_or_tuning": False,
        "cv_splits": cv_splits,
        "models": cv_comparison,
        "selected_model": selected_name,
    }
    write_json(args.report_dir / "model_comparison.json", comparison_payload)
    write_json(args.report_dir / "test_metrics.json", metrics)
    (args.report_dir / "evaluation_report.md").write_text(
        make_evaluation_markdown(
            selected_name,
            cv_comparison,
            metrics,
            matrix.astype(int).tolist(),
            args.input,
            missing_target_rows,
        ),
        encoding="utf-8",
    )

    print(f"EDA report: {args.report_dir / 'eda_report.md'}")
    print(f"Selected using training-only cross-validation: {selected_name}")
    print(f"Fitted pipeline saved to: {args.model_output}")
    print(f"Test metrics saved to: {args.report_dir / 'test_metrics.json'}")
    print(f"Evaluation report saved to: {args.report_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
