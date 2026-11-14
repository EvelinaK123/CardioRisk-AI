# CardioRisk AI — Heart Disease Classification

This is a beginner-friendly machine-learning project for the GitHub repository CardioRisk-AI. It trains a classifier to predict the **HeartDisease label in this CSV dataset**:

- 1 means the dataset label is present.
- 0 means the dataset label is absent.

The model is trained on a binary target and predicts the dataset's HeartDisease label. The prediction output also groups the model's score for label 1 into Low, Medium, and High bands. The CSV does not contain three risk labels, so these bands are applied to the score after training; they are not three classes learned by the model. The cutoffs are project score bands and are not clinically validated risk thresholds.

## Features used

The CSV contains these 11 matching predictor columns:

Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, and ST_Slope.

The project does not use ca (number of major vessels) or thal (perfusion category). Those columns are not in this CSV. They are not fabricated, estimated, or required.

The code uses the category values found in the CSV. Text fields such as Sex and ChestPainType are treated as categorical data and encoded inside a scikit-learn pipeline. Do not manually convert them to numeric codes. The same fitted pipeline is used for training and prediction.

## Run in GitHub Codespaces

1. On GitHub, open this repository and choose Code, then Codespaces, then create or open a Codespace on main.
2. In the Codespace Explorer, make a data folder if it is not there. Upload your private heart.csv file into that folder as data/heart.csv. Keep the original file name.
3. Open the integrated terminal at the repository root and run:

    python -m pip install -r requirements.txt
    python src/eda.py --input data/heart.csv
    python src/train.py --input data/heart.csv --test-size 0.20 --random-state 42

The Codespaces dev-container configuration installs the requirements when a new Codespace is created or rebuilt. The explicit install command above also works in an existing Codespace.

The EDA script writes reports/eda_report.md before training begins. Training compares Logistic Regression and Random Forest using cross-validation on the training portion only. It then selects one model, fits that complete preprocessing-and-model pipeline on the training portion, and evaluates it once on the held-out test portion.

4. To try the prediction script with fictional examples, run:

    python src/predict.py --input data/sample_input_fictional.csv --output reports/sample_predictions.csv

The input CSV for prediction does not need a HeartDisease column. The sample file contains fictional values and has no target column.

To predict from your own unlabeled file, place it in the Codespace and run:

    python src/predict.py --input data/my_new_rows.csv --output reports/my_predictions.csv

## Optional local commands

The primary workflow is Codespaces. If you later choose to run a clone on your own computer, use the commands for your shell from the repository root.

### Windows PowerShell

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python src\eda.py --input data\heart.csv
    python src\train.py --input data\heart.csv --test-size 0.20 --random-state 42
    python src\predict.py --input data\sample_input_fictional.csv --output reports\sample_predictions.csv

### macOS or Linux

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python src/eda.py --input data/heart.csv
    python src/train.py --input data/heart.csv --test-size 0.20 --random-state 42
    python src/predict.py --input data/sample_input_fictional.csv --output reports/sample_predictions.csv

## What the scripts do

- src/eda.py checks the expected column names, data types, missing values, exact duplicate rows, target balance, observed category values, numeric summaries, and suspicious values. It reports zero values in Cholesterol and zero values in RestingBP separately.
- src/train.py creates a stratified train/test split. Stratified means both portions keep roughly the same class balance as the source data. It compares the two models with cross-validation using only the training portion, selects by mean training cross-validation balanced accuracy, fits the selected pipeline on all training rows, and evaluates on the held-out test rows.
- src/predict.py loads the saved pipeline, checks that the 11 predictor columns are present, predicts the dataset label, and writes a results CSV. The output includes the binary class, the score for label 1 when available, and a Low/Medium/High risk_band when the pipeline returns probabilities. Additional input columns are ignored by the model.
- .devcontainer/devcontainer.json configures a Python Codespace and installs the project requirements.
- requirements.txt lists the Python libraries needed by the scripts.
- .gitignore keeps the private training CSV and fitted model out of Git.
- data/sample_input_fictional.csv is a clearly identified fictional example with predictor columns only.
- reports/ and models/ contain placeholders so Git tracks those folders before generated files exist.

## Cleaning decisions

The inspection report records possible data issues but does not silently alter the source data. In particular:

- A zero in Cholesterol or RestingBP is reported as a suspicious value. It is not changed to missing, replaced, or used as a reason to delete a row.
- Duplicate rows are reported and kept.
- Actual missing predictor values are handled inside the training pipeline: numeric values use a median imputer, and categorical values use a most-frequent imputer. These imputers are fitted as part of cross-validation or on training data only.
- Rows without a HeartDisease label cannot be used for supervised training. The EDA report counts them; training excludes only rows whose target label is missing.
- The prediction script does not require a target column and never uses one as a predictor.

## Low, Medium, and High score bands

The prediction CSV adds a `risk_band` column using the model's score for dataset label 1:

- **Low:** score below 0.33
- **Medium:** score from 0.33 up to, but not including, 0.67
- **High:** score of 0.67 or higher

Exactly 0.33 is Medium; exactly 0.67 is High. The model itself still learns the two available target classes, 0 and 1. These three bands are applied afterward to organize the model score; the dataset contains no Low/Medium/High training labels. `predicted_HeartDisease` is still the binary class produced by the model's standard 0.50 cutoff, so it can differ from the three-band label. The cutoffs are simple project score bands, not clinically validated risk thresholds.

## Reading the evaluation

Training writes reports/model_comparison.json, reports/test_metrics.json, and reports/evaluation_report.md. It also saves the selected full pipeline at models/heart_disease_pipeline.joblib.

The model comparison report contains training cross-validation results for both candidates. The final test metrics are for the selected model and the held-out test set.

- Accuracy is the fraction of test rows classified correctly.
- Balanced accuracy averages recall across the two labels, so it is useful when the labels are not evenly represented.
- Precision answers: among rows predicted as label 1, what fraction actually have dataset label 1?
- Recall or sensitivity answers: among rows whose dataset label is 1, what fraction were found?
- Specificity answers: among rows whose dataset label is 0, what fraction were found?
- F1 combines precision and recall.
- ROC-AUC measures ranking across thresholds and is reported when both test labels are present.
- The confusion matrix is ordered as actual labels 0 then 1, with predicted labels 0 then 1. Its cells are true negatives, false positives, false negatives, and true positives.

For this dataset, a false positive is a row with label 0 that the model predicts as 1. A false negative is a row with label 1 that the model predicts as 0. These describe classification errors against this dataset’s labels; they are not clinical conclusions. A high score does not prove clinical usefulness.

## Keeping data and model private

Do not add data/heart.csv or models/heart_disease_pipeline.joblib to a commit. .gitignore excludes them by default. The generated EDA and evaluation reports contain aggregate summaries, not copied source rows. Review any generated file before choosing to publish it.

The dataset and model are not included in this repository. The Codespace stores files in its own workspace; be careful not to commit private files when synchronizing changes.
