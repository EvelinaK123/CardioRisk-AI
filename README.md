# CardioRisk AI — Heart Disease Classification

CardioRisk AI is a prototype machine-learning model that uses 11 checkup measurements to predict a patient’s risk level for heart diseases. This project is a prototype and is not medical advice!

The prediction output groups the model's score for label 1 into Low, Medium, and High bands. The CSV does not contain three risk labels, so these bands are applied to the score after training; they are not three classes learned by the model. The cutoffs are project score bands and are not clinically validated risk thresholds.

## Features used

The CSV contains these 11 matching predictor columns:

Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, and ST_Slope.

The code uses the category values found in the CSV. Text fields such as Sex and ChestPainType are treated as categorical data and encoded inside a scikit-learn pipeline. The same fitted pipeline is used for training and prediction.

## Run locally (Windows - PowerShell)

1. Open the repository, click the **Code** button, and download the ZIP file.
2. Open the ZIP file and extract the `CardioRisk-AI-main` folder.
3. Click the address bar, type `powershell`, and press Enter. The prompt should end with `CardioRisk-AI-main>`. If it says only `PS C:\>`, PowerShell is in the wrong folder.
4. The model was tested with Python 3.14. If you do not have it and use the Python install manager, type this command first:

```powershell
py install 3.14
```

5. Check the Python version:

```powershell
py -3.14 --version
```

6. Run the following commands one at a time, in order, to create the environment and install the packages:

```powershell
py -3.14 -m venv .venv-local314
```

```powershell
.\.venv-local314\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv-local314\Scripts\python.exe -m pip install -r requirements.txt
```

7. Check that the saved model file is present:

```powershell
Test-Path .\models\heart_disease_pipeline.joblib
```

8. To make a prediction, try the existing fictional example. This is one command:

```powershell
.\.venv-local314\Scripts\python.exe src\predict.py --input data\sample_input_fictional.csv --output reports\sample_predictions.csv
```

9. To see the prediction, open `reports/sample_predictions.csv` in the project folder.

## Run locally (Linux or macOS)

1. Open the repository, choose **Code**, and download the ZIP file.
2. Extract the ZIP file to get the `CardioRisk-AI-main` folder.
3. Open Terminal in that folder. The current directory should be `CardioRisk-AI-main`.
4. Use Python 3.14. If you use Python 3.12 instead, replace `python3.14` below with `python3.12`. If Python 3.14 is not installed, install it for your operating system first.
5. Run each command one at a time to create the environment and install the packages:

```bash
python3.14 --version
```

```bash
python3.14 -m venv .venv-local314
```

```bash
source .venv-local314/bin/activate
```

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

6. Check that the saved model file is present:

```bash
test -f models/heart_disease_pipeline.joblib && echo "Model file found"
```

7. To make a prediction, try the existing fictional example. This is one command:

```bash
python src/predict.py --input data/sample_input_fictional.csv --output reports/sample_predictions.csv
```

8. Open `reports/sample_predictions.csv` in the project folder to see the prediction.

## Predict with your own CSV

Create `data/my_new_rows.csv` with these 11 predictor columns and no `HeartDisease` column:

```csv
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
```

Run the command for your operating system:

**Windows PowerShell**

```powershell
.\.venv-local314\Scripts\python.exe src\predict.py --input data\my_new_rows.csv --output reports\my_predictions.csv
```

**Linux or macOS**

```bash
python src/predict.py --input data/my_new_rows.csv --output reports/my_predictions.csv
```

Open `reports/my_predictions.csv` to see the predictions.

## Run in the app

[CardioRisk AI web app](https://cardiorisk-ai-predictionmodel.streamlit.app/#cardio-risk-ai)

## What the scripts do

- src/eda.py checks the expected column names, data types, missing values, exact duplicate rows, target balance, observed category values, numeric summaries, and suspicious values. It reports zero values in Cholesterol and zero values in RestingBP separately.
- src/train.py creates a train/test split. Both portions keep roughly the same class balance as the source data. It compares the two models with cross-validation using only the training portion, selects by mean training cross-validation balanced accuracy, fits the selected pipeline on all training rows, and evaluates on the held-out test rows.
- src/predict.py loads the saved pipeline, checks that the 11 predictor columns are present, predicts the dataset label, and writes a results CSV. The output includes the binary class, the score for label 1 when available, and a Low/Medium/High risk_band when the pipeline returns probabilities. Additional input columns are ignored by the model.
- .devcontainer/devcontainer.json configures a Python Codespace and installs the project requirements.
- requirements.txt lists the Python libraries needed by the scripts.
- .gitignore excludes local training CSVs and newly generated model files. The repository includes a fitted pipeline for prediction.
- data/sample_input_fictional.csv is a clearly identified fictional example with predictor columns only.
- models/ contains the fitted pipeline used for prediction. reports/ stores generated EDA and evaluation files.

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


## Model evaluation and errors
Before training, the EDA report checked column names, data types, missing values, duplicate rows, target balance, category values, and suspicious values. The CSV had 918 rows: 410 with HeartDisease label 0 and 508 with label 1. No exact duplicate rows were found. The report noted 172 zero values in Cholesterol and one zero in RestingBP; these values were kept unchanged.
The model comparison used cross-validation on the training portion only. Random Forest had a mean cross-validation balanced accuracy of 0.850 (SD 0.046), compared with 0.847 (SD 0.049) for Logistic Regression, so Random Forest was selected. On the held-out test set of 184 rows, the selected model had 0.908 accuracy, 0.904 balanced accuracy, 0.897 precision, 0.941 recall/sensitivity, 0.866 specificity, 0.919 F1, and 0.933 ROC-AUC. A majority-class baseline that predicts label 1 for every row would have 0.554 accuracy and 0.500 balanced accuracy on this test split.
The confusion matrix, with actual labels 0 and 1 as rows and predicted labels 0 and 1 as columns, was [[71, 11], [6, 96]]. This means there were 11 false positives—rows with dataset label 0 predicted as 1—and 6 false negatives—rows with dataset label 1 predicted as 0. False positives and false negatives may have occurred because some rows with different HeartDisease labels had similar values across the 11 predictors, making them difficult for the model to distinguish.
The model handles numbers and categories the same way during training and prediction. The risk_band column groups the score for label 1 into Low (<0.33), Medium (0.33 to below 0.67), and High (0.67 or above). These are project score bands; the model was trained on the two HeartDisease labels, not three risk categories. We have not checked probability calibration, performance across age or sex groups, or performance on a separate dataset. A high score on this test split does not establish clinical usefulness. 






