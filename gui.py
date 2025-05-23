"""
Desktop GUI – three tabs: Train summary, Predict, Clusters.
Assumes models are pre-trained by preprocess_and_train.py
"""

import PySimpleGUI as sg
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_DIR = Path("models")
MODELS = ["logreg", "rf", "lgbm"]

# Load metrics table
metrics = joblib.load(MODEL_DIR / "metrics.joblib")

# Tab 1 – metrics overview
metric_layout = [
    [sg.Text("Cross-validated AUC scores")],
]
for m in MODELS:
    auc = metrics[m]["AUC_CV_mean"]
    metric_layout.append([sg.Text(f"{m.upper()}: {auc:.3f}")])

tab1 = [[sg.Column(metric_layout)]]

# Tab 2 – prediction form
form_cols = [
    sg.Text("Enter new student data below then press Predict"),
]
# Dynamically build form from one sample row
sample_row = pd.read_csv("student_depression_dataset.csv").drop(columns=["id", "Depression"]).iloc[0]
inputs = {}
for col, val in sample_row.items():
    key = f"IN_{col}"
    inputs[col] = key
    form_cols.append([sg.Text(col), sg.Input(str(val), key=key)])

form_cols.append([sg.Button("Predict", key="PREDICT"), sg.Text("", key="OUT")])
tab2 = [[sg.Column(form_cols, scrollable=True, vertical_scroll_only=True, size=(400, 500))]]

# Tab 3 – placeholder for cluster plot
tab3 = [[sg.Text("Cluster visual coming soon")]]

layout = [
    [sg.TabGroup([[sg.Tab("Metrics", tab1), sg.Tab("Predict", tab2), sg.Tab("Clusters", tab3)]])]
]

window = sg.Window("Student Depression Classifier", layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    if event == "PREDICT":
        # Gather input as single-row DataFrame
        new_data = pd.DataFrame({col: [values[key]] for col, key in inputs.items()})
        loaded_model = joblib.load(MODEL_DIR / "rf.joblib")  # default Random Forest
        prob = loaded_model.predict_proba(new_data)[0, 1]
        window["OUT"].update(f"Probability of depression: {prob:.2%}")

        # Local SHAP explanation
        explainer = joblib.load(MODEL_DIR / "rf_explainer.joblib")
        shap_values = explainer.shap_values(
            loaded_model.named_steps["prep"].transform(new_data)
        )
        shap.initjs()
        plt.close("all")
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1],
            matplotlib=True,
            show=False,
            figsize=(8, 2),
        )
        plt.savefig("shap_single.png", bbox_inches="tight")
        sg.popup_no_buttons("Local explanation", image="shap_single.png", auto_close=True)

window.close()
