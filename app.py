"""Streamlit interface for the CardioRisk AI dataset-classification model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"

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

LOW_SCORE_MAX = 0.33
HIGH_SCORE_MIN = 0.67

CATEGORY_LABELS = {
    "Sex": {
        "F": "Female (F)",
        "M": "Male (M)",
    },
    "ChestPainType": {
        "ASY": "Asymptomatic (ASY)",
        "ATA": "Atypical angina (ATA)",
        "NAP": "Non-anginal pain (NAP)",
        "TA": "Typical angina (TA)",
    },
    "RestingECG": {
        "LVH": "LVH (left ventricular hypertrophy)",
        "Normal": "Normal",
        "ST": "ST-T wave abnormality (ST)",
    },
    "ExerciseAngina": {
        "N": "No (N)",
        "Y": "Yes (Y)",
    },
    "ST_Slope": {
        "Down": "Down",
        "Flat": "Flat",
        "Up": "Up",
    },
}

PROTOTYPE_WARNING = (
    "Prototype only: this is not a certified medical tool and is not medical advice. "
    "Do not use its output to make health-care decisions."
)

st.set_page_config(
    page_title="CardioRisk AI | Heart Disease Classification",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --cardio-pink: #fab9bd;
        --cardio-red: #a92132;
        --cardio-deep-red: #861a29;
        --cardio-soft-pink: #fff2f3;
        --cardio-text: #3f2529;
    }
    .stApp {
        background: #ffffff;
        color: var(--cardio-text);
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: var(--cardio-soft-pink);
        border-right: 1px solid var(--cardio-pink);
    }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        background: linear-gradient(115deg, var(--cardio-deep-red), #bd3445);
        color: #ffffff;
        margin-bottom: 1.1rem;
        box-shadow: 0 12px 32px rgba(134, 26, 41, 0.16);
        border-bottom: 5px solid var(--cardio-pink);
    }
    .hero h1 {
        color: #ffffff;
        margin: 0.25rem 0 0.35rem 0;
        font-size: 2.25rem;
    }
    .hero p {
        color: #fff0f1;
        margin: 0;
        font-size: 1.05rem;
    }
    .eyebrow {
        color: #ffdadd;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    h2, h3 {
        color: var(--cardio-deep-red);
    }
    .info-card {
        background: #ffffff;
        border: 1px solid var(--cardio-pink);
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        min-height: 112px;
        box-shadow: 0 4px 14px rgba(169, 33, 50, 0.06);
    }
    .info-card strong {
        display: block;
        color: var(--cardio-red);
        font-size: 1.45rem;
        margin-bottom: 0.3rem;
    }
    .info-card span {
        color: #62484b;
        font-size: 0.92rem;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid var(--cardio-pink);
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin: 0.55rem 0 1rem 0;
    }
    .risk-card {
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        margin: 0.8rem 0 1rem 0;
        border: 1px solid var(--cardio-pink);
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(169, 33, 50, 0.06);
    }
    .risk-card h2 {
        margin: 0.15rem 0 0.35rem 0;
        font-size: 2rem;
    }
    .risk-low { color: #16705d; }
    .risk-medium { color: #9b6500; }
    .risk-high { color: var(--cardio-red); }
    .small-label {
        color: #70484d;
        font-size: 0.83rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: var(--cardio-red);
        border-color: var(--cardio-red);
        color: #ffffff;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: var(--cardio-deep-red);
        border-color: var(--cardio-deep-red);
    }
    [data-testid="stAlert"],
    [data-testid="stAlertContainer"],
    [data-testid^="stAlertContent"] {
        background: var(--cardio-soft-pink) !important;
        border-color: var(--cardio-pink) !important;
        color: var(--cardio-deep-red) !important;
    }
    [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {
        color: var(--cardio-deep-red) !important;
    }
    [data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background: #ffffff;
        border-color: var(--cardio-pink);
    }
    input:focus,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--cardio-red);
        box-shadow: 0 0 0 0.15rem rgba(250, 185, 189, 0.42);
    }
    div[data-testid="stMetricValue"] {
        color: var(--cardio-red);
    }
    a {
        color: var(--cardio-red);
    }
    a:hover {
        color: var(--cardio-deep-red);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Checkup data classification demo</div>
        <h1>CardioRisk AI</h1>
        <p>A transparent, educational interface for the HeartDisease dataset model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.warning(PROTOTYPE_WARNING)

with st.sidebar:
    st.markdown("### 🫀 CardioRisk AI")
    st.caption("Heart disease dataset classification")
    page = st.radio(
        "Navigate",
        ["Project Overview", "Prediction"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "This prototype predicts a dataset label. Its score bands have not been "
        "validated for real-world health decisions."
    )


@st.cache_resource
def load_pipeline(model_path: str):
    """Load the fitted preprocessing-and-model pipeline once per app session."""
    return joblib.load(model_path)


def score_band(score: float) -> str:
    """Turn the class-1 model score into the agreed project bands."""
    if score < LOW_SCORE_MAX:
        return "Low"
    if score < HIGH_SCORE_MIN:
        return "Medium"
    return "High"


def render_overview() -> None:
    st.subheader("Project Overview")
    st.write(
        "CardioRisk AI is a prototype machine-learning project. It uses 11 "
        "checkup-style measurements to predict the HeartDisease label found in "
        "the training dataset: 0 means the label is absent and 1 means it is present."
    )
    st.write(
        "The model is trained on two dataset labels. The Low, Medium, and High "
        "levels shown by this interface are score bands applied after training; "
        "they are not three classes learned by the model."
    )

    card1, card2, card3 = st.columns(3)
    with card1:
        st.markdown(
            '<div class="info-card"><strong>11 features</strong>'
            "<span>Available predictor fields used by the model</span></div>",
            unsafe_allow_html=True,
        )
    with card2:
        st.markdown(
            '<div class="info-card"><strong>2 labels</strong>'
            "<span>HeartDisease: 0 or 1 in the dataset</span></div>",
            unsafe_allow_html=True,
        )
    with card3:
        st.markdown(
            '<div class="info-card"><strong>3 score bands</strong>'
            "<span>Low, Medium, and High using project cutoffs</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Real-world problem this explores")
    st.write(
        "The project explores whether routinely recorded checkup data can help "
        "organize example records for further review. This prototype has not been "
        "validated to prioritize real people or support medical decisions."
    )

    st.markdown("### Goals")
    st.markdown(
        """
        - Show how a classification model can use numeric and text-category fields together.
        - Reuse the same saved preprocessing-and-model pipeline for predictions.
        - Present the model's dataset-label score in simple Low, Medium, and High bands.
        - Make the model's limits clear alongside its output.
        """
    )

    st.markdown("### Dataset credits")
    st.write(
        "The private heart.csv file was supplied by the project owner and is not "
        "included in this repository. Its 918-row schema appears to match the "
        "Heart Failure Prediction Dataset by fedesoriano on Kaggle. Please confirm "
        "that this is the source of your copy and check its dataset page for the "
        "applicable license before redistributing the data."
    )
    st.markdown(
        "[Open the Heart Failure Prediction Dataset page on Kaggle]"
        "(https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)"
    )

    st.markdown("### What the output means")
    st.write(
        "The prediction page shows a Low, Medium, or High band based on the "
        "model's score for dataset label 1. It also lets you expand the result "
        "to see the predicted binary dataset label and the model score."
    )
    st.markdown(
        """
        - **Low:** score below 0.33
        - **Medium:** score from 0.33 up to, but not including, 0.67
        - **High:** score of 0.67 or higher
        """
    )
    st.info(
        "The score is not a confirmed real-world probability. These cutoffs are "
        "simple project defaults and are not medically validated."
    )

    with st.expander("A short guide to the terms"):
        st.markdown(
            """
            - **ChestPainType:** the dataset's category for chest-pain presentation:
              TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), or
              ASY (asymptomatic).
            - **RestingBP:** resting blood pressure recorded in the dataset, in mmHg.
            - **Cholesterol:** the dataset's serum cholesterol measurement, in mg/dL.
            - **FastingBS:** a dataset indicator for fasting blood sugar above
              120 mg/dL (1 = yes, 0 = no).
            - **RestingECG:** the dataset's resting ECG category. LVH means
              left ventricular hypertrophy; ST is the dataset's ST-T abnormality
              category; Normal is the normal category.
            - **MaxHR:** the maximum heart rate recorded during the exercise test.
            - **ExerciseAngina:** whether exercise-induced angina was recorded.
            - **Oldpeak:** ST depression during exercise compared with rest.
            - **ST_Slope:** the Up, Flat, or Down category for the peak exercise
              ST segment slope.
            """
        )
        st.caption(
            "These short descriptions explain dataset fields; they are not "
            "interpretations of an individual's health."
        )


def render_prediction() -> None:
    st.subheader("Prediction")
    st.write(
        "Enter one example row using the 11 available fields. The form starts "
        "with fictional example values; change them to match the row you want to try."
    )
    st.caption(
        "The form does not ask for HeartDisease because that is the target the "
        "model predicts. It is not an input feature."
    )

    if not MODEL_PATH.is_file():
        st.error("The saved model file is not available in this Codespace yet.")
        st.write(
            "The model file is kept out of GitHub. In the Codespace terminal, "
            "upload heart.csv to the data folder if needed, then train the existing model:"
        )
        st.code(
            "python src/train.py --input data/heart.csv "
            "--test-size 0.20 --random-state 42",
            language="bash",
        )
        st.stop()

    try:
        pipeline = load_pipeline(str(MODEL_PATH))
    except Exception as exc:
        st.error(
            "The saved model could not be loaded. Check the Python package versions "
            "or retrain it in this Codespace with the project's training command."
        )
        with st.expander("Technical detail"):
            st.code(str(exc))
        st.stop()

    with st.form("prediction_form"):
        st.markdown("#### 1. Numeric measurements")
        first_row = st.columns(3)
        with first_row[0]:
            age = st.number_input(
                "Age (years)", min_value=0, max_value=120, value=52, step=1
            )
        with first_row[1]:
            resting_bp = st.number_input(
                "RestingBP (mmHg)", min_value=0, max_value=300, value=130, step=1
            )
        with first_row[2]:
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)", min_value=0, max_value=1000, value=220, step=1
            )

        second_row = st.columns(3)
        with second_row[0]:
            fasting_bs = st.selectbox(
                "Fasting blood sugar above 120 mg/dL",
                options=[0, 1],
                format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
            )
        with second_row[1]:
            max_hr = st.number_input(
                "MaxHR (beats per minute)",
                min_value=0,
                max_value=250,
                value=160,
                step=1,
            )
        with second_row[2]:
            oldpeak = st.number_input(
                "Oldpeak",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                format="%.1f",
                help="Dataset measurement of exercise-related ST depression compared with rest.",
            )

        st.markdown("#### 2. Category choices")
        third_row = st.columns(3)
        with third_row[0]:
            sex = st.selectbox(
                "Sex",
                options=list(CATEGORY_LABELS["Sex"]),
                format_func=lambda value: CATEGORY_LABELS["Sex"][value],
            )
        with third_row[1]:
            chest_pain = st.selectbox(
                "ChestPainType",
                options=list(CATEGORY_LABELS["ChestPainType"]),
                format_func=lambda value: CATEGORY_LABELS["ChestPainType"][value],
            )
        with third_row[2]:
            resting_ecg = st.selectbox(
                "RestingECG",
                options=list(CATEGORY_LABELS["RestingECG"]),
                format_func=lambda value: CATEGORY_LABELS["RestingECG"][value],
            )

        fourth_row = st.columns(2)
        with fourth_row[0]:
            exercise_angina = st.selectbox(
                "ExerciseAngina",
                options=list(CATEGORY_LABELS["ExerciseAngina"]),
                format_func=lambda value: CATEGORY_LABELS["ExerciseAngina"][value],
            )
        with fourth_row[1]:
            st_slope = st.selectbox(
                "ST_Slope",
                options=list(CATEGORY_LABELS["ST_Slope"]),
                format_func=lambda value: CATEGORY_LABELS["ST_Slope"][value],
            )

        submitted = st.form_submit_button(
            "Generate prototype result",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "The app does not write form entries to a CSV or database. Avoid entering "
        "names, contact details, or other direct identifiers."
    )

    if not submitted:
        return

    input_row = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    if resting_bp == 0 or cholesterol == 0:
        st.info(
            "A zero was entered for RestingBP or Cholesterol. Zero values appeared "
            "in the training CSV and were kept unchanged; check that the input is intentional."
        )

    try:
        predicted_label = int(pipeline.predict(input_row)[0])
        classifier = pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        if 1 not in classes or not hasattr(pipeline, "predict_proba"):
            st.error(
                "This saved model does not provide the class-1 score needed for "
                "the Low/Medium/High display."
            )
            return
        score = float(pipeline.predict_proba(input_row)[0, classes.index(1)])
    except Exception as exc:
        st.error(
            "The model could not process this row. The category values must match "
            "the dataset labels shown in the form."
        )
        with st.expander("Technical detail"):
            st.code(str(exc))
        return

    band = score_band(score)
    band_style = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }[band]

    st.markdown("### Result")
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="small-label">Model score band</div>
            <h2 class="{band_style}">{band}</h2>
            <div>Based on the model's score for dataset label 1.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_column, band_column = st.columns(2)
    with score_column:
        st.metric("Model score for dataset label 1", f"{score:.3f}")
    with band_column:
        st.metric("Project score band", band)

    st.caption(
        "This score is produced by the fitted classifier and is not a calibrated "
        "or validated real-world probability."
    )

    with st.expander("Show technical result"):
        st.write(
            f"Predicted HeartDisease dataset label: {predicted_label} "
            f"({'present' if predicted_label == 1 else 'absent'} in the dataset)."
        )
        st.write(
            "The model's standard binary class uses its default 0.50 decision "
            "cutoff. The Low/Medium/High band is assigned separately using 0.33 "
            "and 0.67, so the two outputs can differ."
        )

    st.warning(PROTOTYPE_WARNING)


if page == "Project Overview":
    render_overview()
else:
    render_prediction()
