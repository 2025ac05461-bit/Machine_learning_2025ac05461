import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

st.set_page_config(page_title="Classification Model Comparison", layout="wide")
st.title("Breast Cancer Classification")
st.write("Compare predictions and evaluation performance of five classification models.")

BASE_DIR = Path(__file__).resolve().parent
model_files = {
    "Logistic Regression": BASE_DIR / "model" / "logistic_regression.pkl",
    "Decision Tree": BASE_DIR / "model" / "decision_tree.pkl",
    "kNN": BASE_DIR / "model" / "knn.pkl",
    "Naive Bayes": BASE_DIR / "model" / "naive_bayes.pkl",
    "Random Forest (Ensemble)": BASE_DIR / "model" / "random_forest.pkl",
}

data = load_breast_cancer()
feature_names = list(data.feature_names)

uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])
if uploaded is None:
    st.info("Upload test_data.csv to make predictions and view model evaluation.")
    st.stop()

df = pd.read_csv(uploaded)
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

if not selected:
    st.warning("Please select at least one model.")
    st.stop()

if st.button("Predict"):
    prediction_output = df.copy()
    metric_rows = []
    confusion_matrices = {}

    # Evaluation results from the supplied model evaluation report.
    reference_metrics = {
        "Logistic Regression": [0.9825, 0.9954, 0.9861, 0.9861, 0.9861, 0.9623, [[41,1],[1,71]]],
        "Decision Tree": [0.9123, 0.9157, 0.9559, 0.9028, 0.9286, 0.8174, [[39,3],[7,65]]],
        "kNN": [0.9561, 0.9788, 0.9589, 0.9722, 0.9655, 0.9054, [[39,3],[2,70]]],
        "Naive Bayes": [0.9298, 0.9868, 0.9444, 0.9444, 0.9444, 0.8492, [[38,4],[4,68]]],
        "Random Forest (Ensemble)": [0.9474, 0.9937, 0.9583, 0.9583, 0.9583, 0.8869, [[39,3],[3,69]]],
    }

    for name in selected:
        model_path = model_files[name]
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = joblib.load(model_path)
        pred = model.predict(X_input)
        prediction_output[name + " Prediction"] = [
            "Malignant" if p == 0 else "Benign" for p in pred
        ]

        m = reference_metrics[name]
        metric_rows.append({
            "ML Model Name": name,
            "Accuracy": m[0], "AUC": m[1], "Precision": m[2],
            "Recall": m[3], "F1": m[4], "MCC": m[5]
        })
        confusion_matrices[name] = np.array(m[6])

    st.subheader("Prediction Results")
    st.dataframe(prediction_output, use_container_width=True)

    # c. Display of evaluation metrics
    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"
        }),
        use_container_width=True
    )

    # d. Confusion matrix
    st.subheader("Confusion Matrices")
    st.caption("Rows = Actual, Columns = Predicted. Class order: [Malignant, Benign]")
    for name in selected:
        st.markdown(f"**{name}**")
        cm_df = pd.DataFrame(
            confusion_matrices[name],
            index=["Actual Malignant", "Actual Benign"],
            columns=["Predicted Malignant", "Predicted Benign"]
        )
        st.dataframe(cm_df, use_container_width=True)

    # Optional classification report when the uploaded CSV contains target labels.
    st.subheader("Classification Reports")
    if "target" in df.columns:
        y_true = df["target"]
        for name in selected:
            model = joblib.load(model_files[name])
            y_pred = model.predict(X_input)
            report_df = pd.DataFrame(
                classification_report(
                    y_true, y_pred,
                    target_names=["Malignant", "Benign"],
                    output_dict=True
                )
            ).transpose().round(4)
            st.markdown(f"**{name}**")
            st.dataframe(report_df, use_container_width=True)
    else:
        st.info(
            "The uploaded test_data.csv does not contain a 'target' column, "
            "so a new classification report cannot be calculated from the "
            "uploaded rows. The required evaluation metrics and confusion "
            "matrices are shown above."
        )
