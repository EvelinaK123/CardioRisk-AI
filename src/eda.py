"""Inspect the CSV and write a beginner-friendly EDA report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
TARGET_COLUMN = "HeartDisease"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
NUMERIC_CHECKS = {
    "Age": (0, 120),
    "RestingBP": (0, 300),
    "Cholesterol": (0, 1000),
    "MaxHR": (0, 250),
    "Oldpeak": (-10, 10),
}


def _value_label(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)


def _markdown_table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for left, right in rows:
        safe_left = left.replace("|", "\\|").replace("\n", " ")
        safe_right = right.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {safe_left} | {safe_right} |")
    return "\n".join(lines)


def build_eda_report(frame: pd.DataFrame, input_name: str = "input CSV") -> str:
    """Create a report without modifying or cleaning the supplied rows."""
    columns = [str(name) for name in frame.columns]
    missing_columns = [name for name in EXPECTED_COLUMNS if name not in columns]
    extra_columns = [name for name in columns if name not in EXPECTED_COLUMNS]

    lines = [
        "# Data inspection and EDA report",
        "",
        f"Input: {input_name}",
        "",
        "## Dataset shape and schema",
        "",
        f"- Rows: {len(frame)}",
        f"- Columns: {len(frame.columns)}",
        f"- Column names: {', '.join(columns) if columns else '(none)'}",
        f"- Required project columns missing: {', '.join(missing_columns) if missing_columns else 'none'}",
        f"- Additional columns: {', '.join(extra_columns) if extra_columns else 'none'}",
        "",
        "The model uses only the 11 listed predictor columns and HeartDisease as its target. "
        "It does not use ca or thal.",
        "",
        "### Data types",
        "",
        _markdown_table(
            [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()],
            ("Column", "Pandas data type"),
        ),
        "",
        "## Missing values",
        "",
        _markdown_table(
            [
                (str(column), str(int(count)))
                for column, count in frame.isna().sum().items()
            ],
            ("Column", "Missing cells"),
        ),
        "",
        "## Duplicate rows",
        "",
        f"- Exact duplicate rows: {int(frame.duplicated().sum())}",
        "- Duplicate rows are reported and kept; this script does not remove them.",
        "",
        "## Target balance",
        "",
    ]

    if TARGET_COLUMN in frame.columns:
        counts = frame[TARGET_COLUMN].value_counts(dropna=False, sort=False)
        total = len(frame)
        balance_rows = []
        for value, count in counts.items():
            percent = (100.0 * int(count) / total) if total else 0.0
            balance_rows.append((_value_label(value), f"{int(count)} ({percent:.2f}%)"))
        lines.append(_markdown_table(balance_rows, ("HeartDisease dataset label", "Rows")))
        lines.append("")
        lines.append("- Dataset label 1 means present; dataset label 0 means absent.")
        invalid_target = frame[TARGET_COLUMN].notna() & ~frame[TARGET_COLUMN].isin([0, 1])
        lines.append(
            f"- Non-missing target values outside 0 and 1: {int(invalid_target.sum())}"
        )
        lines.append(
            f"- Rows with missing target labels: {int(frame[TARGET_COLUMN].isna().sum())}"
        )
    else:
        lines.append("The HeartDisease target column is missing, so target balance cannot be calculated.")
    lines.extend(["", "## Observed category values", ""])

    category_rows = []
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            category_rows.append((column, "column missing"))
            continue
        counts = frame[column].value_counts(dropna=False, sort=False)
        observed = ", ".join(
            f"{_value_label(value)}: {int(count)}" for value, count in counts.items()
        )
        category_rows.append((column, observed or "(no rows)"))
    lines.append(_markdown_table(category_rows, ("Categorical column", "Observed values and counts")))
    lines.extend(["", "These are the CSV's actual values; the project does not assume numeric codes.", ""])

    lines.extend(["## Numeric summaries", ""])
    numeric_rows = []
    for column in ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]:
        if column not in frame.columns:
            numeric_rows.append((column, "column missing"))
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            summary = (
                f"min={values.min():g}; median={values.median():g}; "
                f"mean={values.mean():.3f}; max={values.max():g}"
            )
        else:
            summary = "no numeric values"
        numeric_rows.append((column, summary))
    lines.append(_markdown_table(numeric_rows, ("Numeric column", "Summary")))
    lines.extend(["", "## Suspicious values to review", ""])

    suspicious_rows = []
    for column, (low, high) in NUMERIC_CHECKS.items():
        if column not in frame.columns:
            suspicious_rows.append((column, "column missing"))
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if column == "RestingBP":
            zero_count = int((values == 0).sum())
            suspicious_rows.append((column, f"zero values: {zero_count}; values below 0 or above {high}: {int(((values < 0) | (values > high)).sum())}"))
        elif column == "Cholesterol":
            zero_count = int((values == 0).sum())
            suspicious_rows.append((column, f"zero values: {zero_count}; values below 0 or above {high}: {int(((values < 0) | (values > high)).sum())}"))
        else:
            out_of_range = int(((values < low) | (values > high)).sum())
            suspicious_rows.append((column, f"values outside {low} to {high}: {out_of_range}"))

    if "FastingBS" in frame.columns:
        fasting = pd.to_numeric(frame["FastingBS"], errors="coerce")
        suspicious_rows.append(("FastingBS", f"values other than 0 or 1: {int((fasting.notna() & ~fasting.isin([0, 1])).sum())}"))
    if TARGET_COLUMN in frame.columns:
        target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        suspicious_rows.append(("HeartDisease", f"non-missing values other than 0 or 1: {int((target.notna() & ~target.isin([0, 1])).sum())}"))

    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            continue
        text_values = frame[column].dropna().astype(str)
        whitespace_count = int((text_values != text_values.str.strip()).sum())
        suspicious_rows.append((column, f"values with leading or trailing whitespace: {whitespace_count}"))

    lines.append(_markdown_table(suspicious_rows, ("Field", "Check result")))
    lines.extend(
        [
            "",
            "## Cleaning decisions",
            "",
            "- This report only inspects the data. It does not rewrite values, delete duplicate rows, or drop rows because a numeric value is zero.",
            "- Zero Cholesterol and zero RestingBP values are counted and shown above. They remain numeric observations; they are not silently treated as missing.",
            "- During modeling, actual missing predictor cells are imputed inside the scikit-learn pipeline. Numeric imputers use the training median; categorical imputers use the training most-frequent value.",
            "- Rows with a missing target cannot be used for supervised training. The training script excludes only those unlabeled rows and reports their count.",
            "- The training script rejects target values other than 0 and 1.",
            "",
        ]
    )
    return "\n".join(lines)


def write_eda_report(frame: pd.DataFrame, output_path: Path, input_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_eda_report(frame, input_name=input_name),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/heart.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/eda_report.md"))
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"CSV not found: {args.input}. Put the private dataset at data/heart.csv "
            "or pass its path with --input."
        )

    frame = pd.read_csv(args.input)
    write_eda_report(frame, args.output, str(args.input))
    print(f"EDA report written to {args.output}")
    print(f"Rows: {len(frame)}; columns: {len(frame.columns)}")
    print(f"Columns found: {', '.join(str(name) for name in frame.columns)}")


if __name__ == "__main__":
    main()
