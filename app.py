import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Classification Model Comparison", layout="wide")
st.title("Breast Cancer Classification")
st.write("Compare predictions from five trained classification models.")

data = load_breast_cancer()
feature_names = list(data.feature_names)

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}

uploaded = st.file_uploader("Upload a CSV containing the 30 model features", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload test_data.csv to make predictions.")
    st.stop()

missing = [c for c in feature_names if c not in df.columns]
if missing:
    st.error(f"The uploaded CSV is missing {len(missing)} required feature columns.")
    st.write(missing)
    st.stop()

X_input = df[feature_names].copy()

selected = st.multiselect(
    "Select models",
    list(model_files.keys()),
    default=list(model_files.keys())
)

if st.button("Predict") and selected:
    output = df.copy()
    for name in selected:
        model = joblib.load(model_files[name])
        pred = model.predict(X_input)
        output[name + " Prediction"] = [
            "Malignant" if p == 0 else "Benign" for p in pred
        ]
    st.subheader("Predictions")
    st.dataframe(output, use_container_width=True)
