# Running the Streamlit app in GitHub Codespaces

This app is a separate interface for the existing fitted model. It does not change
the EDA, training, or command-line prediction scripts.

## Start the app

1. Open the CardioRisk AI repository and open your existing Codespace.
2. In the Codespace terminal, sync the new app files:

   ```bash
   git pull origin main
   ```

3. Install the app dependency:

   ```bash
   python -m pip install -r requirements-app.txt
   ```

4. Confirm that the fitted model exists at
   `models/heart_disease_pipeline.joblib`. The model is private and ignored by Git.
   If it is missing, upload your private `heart.csv` into the Codespace's `data`
   folder, then run the existing training script:

   ```bash
   python src/train.py --input data/heart.csv --test-size 0.20 --random-state 42
   ```

5. Start Streamlit:

   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

6. In the Codespaces **Ports** panel, open port **8501** in your browser.

## App pages

- **Project Overview** describes the project, intended demonstration use, dataset
  credit, output, and common terms in the predictor fields.
- **Prediction** provides one form with number inputs and category choices that
  match the labels used by the dataset.

The app loads the same fitted preprocessing-and-model pipeline saved by
`src/train.py`. Numeric values and text categories are passed to that pipeline
without manually encoding the text values. The app does not save the submitted
form values to a CSV or database.

The Low, Medium, and High bands use the project's agreed cutoffs: below 0.33 is
Low, 0.33 to below 0.67 is Medium, and 0.67 or above is High. The model is still
trained on the binary HeartDisease dataset label; these bands are added to its
score afterward. The score is not a calibrated real-world probability, and the
cutoffs are not clinically validated.

**Prototype only: this is not a certified medical tool and is not medical advice.**
