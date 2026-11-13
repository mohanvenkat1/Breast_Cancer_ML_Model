"""
==============================================================================
  Breast Cancer ANN – Streamlit Web Application
  File   : app.py
  Purpose: Interactive frontend — user enters 30 feature values and receives
           a Benign / Malignant prediction with probability.
==============================================================================
Run with:
    streamlit run app.py
==============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

# ─────────────────────────── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer ANN Classifier",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── Paths ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "breast_cancer_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ─────────────────────────── Feature metadata ───────────────────────────────
# (from Wisconsin Diagnostic Breast Cancer dataset)
FEATURE_GROUPS = {
    "Mean Features (Columns 1–10)": [
        ("mean radius",            "Mean of distances from center to points on perimeter",      6.0,  30.0,  14.1),
        ("mean texture",           "Standard deviation of gray-scale values",                   9.0,  40.0,  19.3),
        ("mean perimeter",         "Mean perimeter of the cell nucleus",                       40.0, 200.0,  92.0),
        ("mean area",              "Mean area of the cell nucleus",                           100.0,2500.0, 655.0),
        ("mean smoothness",        "Mean of local variation in radius lengths",                0.05,  0.17,  0.096),
        ("mean compactness",       "Mean of perimeter² / area - 1.0",                         0.02,  0.35,  0.104),
        ("mean concavity",         "Mean severity of concave portions of the contour",         0.0,   0.43,  0.089),
        ("mean concave points",    "Mean number of concave portions of the contour",           0.0,   0.20,  0.049),
        ("mean symmetry",          "Mean symmetry of the nucleus",                             0.10,  0.30,  0.181),
        ("mean fractal dimension", "Mean coastline approximation - 1",                         0.05,  0.10,  0.063),
    ],
    "SE Features (Columns 11–20)": [
        ("radius error",            "Standard error of distances",                              0.1,   3.0,  0.405),
        ("texture error",           "Standard error of texture",                                0.3,   5.0,  1.22),
        ("perimeter error",         "Standard error of perimeter",                              0.5,  25.0,  2.87),
        ("area error",              "Standard error of area",                                   6.0, 550.0, 40.3),
        ("smoothness error",        "Standard error of smoothness",                             0.001, 0.03, 0.007),
        ("compactness error",       "Standard error of compactness",                            0.002, 0.14, 0.025),
        ("concavity error",         "Standard error of concavity",                              0.0,   0.40, 0.032),
        ("concave points error",    "Standard error of concave points",                         0.0,   0.05, 0.012),
        ("symmetry error",          "Standard error of symmetry",                               0.007, 0.08, 0.021),
        ("fractal dimension error", "Standard error of fractal dimension",                      0.001, 0.03, 0.004),
    ],
    "Worst Features (Columns 21–30)": [
        ("worst radius",            "Largest mean value for radius",                            7.0,  40.0, 16.3),
        ("worst texture",           "Largest mean value for texture",                          12.0,  50.0, 25.7),
        ("worst perimeter",         "Largest mean value for perimeter",                        50.0, 260.0,107.3),
        ("worst area",              "Largest mean value for area",                            185.0,4254.0,880.6),
        ("worst smoothness",        "Largest mean value for smoothness",                        0.07,  0.23,  0.132),
        ("worst compactness",       "Largest mean value for compactness",                       0.03,  1.06,  0.254),
        ("worst concavity",         "Largest mean value for concavity",                         0.0,   1.25,  0.272),
        ("worst concave points",    "Largest mean value for concave points",                    0.0,   0.29,  0.115),
        ("worst symmetry",          "Largest mean value for symmetry",                          0.16,  0.66,  0.290),
        ("worst fractal dimension", "Largest mean value for fractal dimension",                 0.055, 0.21,  0.084),
    ],
}

ALL_FEATURES = []
for features in FEATURE_GROUPS.values():
    for feat in features:
        ALL_FEATURES.append(feat[0])   # name only

# ─────────────────────────── Load Model & Scaler ────────────────────────────
@st.cache_resource(show_spinner="Loading trained ANN model…")
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# ─────────────────────────── Header ─────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#C0392B;'>🎗️ Breast Cancer ANN Classifier</h1>
<p style='text-align:center; color:gray; font-size:16px;'>
Wisconsin Diagnostic Breast Cancer (WDBC) Dataset · Artificial Neural Network · Binary Classification
</p>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────── Sidebar: Model status ──────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.info(
        "This application uses a trained **Artificial Neural Network (ANN)** "
        "to predict whether a breast cancer record is **Benign** or **Malignant** "
        "based on 30 numerical features extracted from digitized FNA images."
    )
    if model is None:
        st.error(
            "⚠️ Model not found!\n\nPlease run the training script first:\n"
            "```\npython src/train.py\n```"
        )
    else:
        st.success("✅ Model loaded successfully")
        st.markdown("**Model Architecture:**")
        st.code(
            "Input(30) → Dense(64, ReLU) → Dropout(0.3)\n"
            "         → Dense(32, ReLU) → Dropout(0.2)\n"
            "         → Dense(1, Sigmoid)",
            language="text"
        )

    st.markdown("---")
    st.markdown("**Prediction threshold:** `0.50`")
    st.markdown("Probability ≥ 0.5 → **Malignant**")
    st.markdown("Probability < 0.5 → **Benign**")

# ─────────────────────────── Main: Input Form ───────────────────────────────
st.markdown("## 🔬 Enter Tumor Feature Values")
st.caption(
    "Fill in the 30 numerical features below (grouped by Mean, SE, and Worst). "
    "Default values are dataset medians."
)

input_values = {}

with st.form("prediction_form"):
    for group_name, features in FEATURE_GROUPS.items():
        st.markdown(f"### {group_name}")
        cols = st.columns(2)
        for idx, (name, description, min_v, max_v, default) in enumerate(features):
            col = cols[idx % 2]
            with col:
                val = st.number_input(
                    label=name,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(default),
                    step=float((max_v - min_v) / 1000),
                    help=description,
                    format="%.5f",
                    key=name
                )
                input_values[name] = val
        st.markdown("---")

    submitted = st.form_submit_button("🔍 Predict Diagnosis", use_container_width=True)

# ─────────────────────────── Prediction & Result ────────────────────────────
if submitted:
    if model is None or scaler is None:
        st.error("Cannot predict — model not loaded. Run `python src/train.py` first.")
    else:
        # Build feature vector in the correct order
        feature_vector = np.array([[input_values[feat] for feat in ALL_FEATURES]])

        # Scale using the saved scaler
        feature_scaled = scaler.transform(feature_vector)

        # Predict
        prob_malignant = float(model.predict(feature_scaled, verbose=0).flatten()[0])
        prob_benign    = 1.0 - prob_malignant
        prediction     = "Malignant" if prob_malignant >= 0.5 else "Benign"

        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == "Malignant":
                st.error(f"### 🔴 {prediction}")
            else:
                st.success(f"### 🟢 {prediction}")

        with col2:
            st.metric("Probability — Malignant", f"{prob_malignant:.4f} ({prob_malignant*100:.2f}%)")

        with col3:
            st.metric("Probability — Benign",    f"{prob_benign:.4f} ({prob_benign*100:.2f}%)")

        # Gauge / progress bar
        st.markdown("#### Malignancy Probability")
        st.progress(prob_malignant)

        if prediction == "Malignant":
            st.warning(
                "⚠️ **High risk of malignancy detected.** This is a computational prediction and "
                "must be confirmed by a qualified medical professional."
            )
        else:
            st.info(
                "ℹ️ **Likely benign.** This is a computational prediction and must be "
                "confirmed by a qualified medical professional."
            )

        # Show entered features
        with st.expander("📋 View entered feature values"):
            df_input = pd.DataFrame([input_values]).T
            df_input.columns = ["Value"]
            df_input.index.name = "Feature"
            st.dataframe(df_input, use_container_width=True)

        st.caption(
            "⚕️ **Medical Disclaimer**: This tool is for educational and research purposes only. "
            "It is NOT a substitute for professional medical diagnosis. Always consult a licensed physician."
        )
