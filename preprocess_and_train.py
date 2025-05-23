"""
preprocess_and_train.py
Loads the student depression CSV, builds one preprocessing pipeline,
trains three classifiers, and saves models plus SHAP explainers.

Run once:
    python preprocess_and_train.py
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn import __version__ as skver
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# LightGBM is optional. If it is not installed, the script skips it.
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not found, skipping that model.")

import shap
import scipy.sparse as sp

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = Path("student_depression_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "Depression"
ID_COLS = ["id"]           # adjust if your file has no id column
RANDOM_STATE = 42
TEST_RATIO = 0.30
N_JOBS = -1

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded data shape: {df.shape}")

X = df.drop(columns=ID_COLS + [TARGET_COL])
y = df[TARGET_COL]

# -----------------------------------------------------------------------------
# 2. Build preprocessing pipeline
# -----------------------------------------------------------------------------
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

major, minor = map(int, skver.split(".")[:2])
if major > 1 or (major == 1 and minor >= 2):
    encoder = OneHotEncoder(handle_unknown="ignore",
                            sparse_output=False,
                            dtype=float)
else:
    encoder = OneHotEncoder(handle_unknown="ignore",
                            sparse=False,
                            dtype=float)

numeric_pipe = Pipeline([("scaler", StandardScaler())])
categorical_pipe = Pipeline([("encoder", encoder)])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ]
)

# -----------------------------------------------------------------------------
# 3. Train-test split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_RATIO,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------------------------------------------------------
# 4. Define model zoo
# -----------------------------------------------------------------------------
models = {
    "logreg": LogisticRegression(max_iter=1000, n_jobs=N_JOBS),
    "rf": RandomForestClassifier(
        n_estimators=400,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE),
}

if HAS_LGBM:
    models["lgbm"] = LGBMClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
    )

# -----------------------------------------------------------------------------
# 5. Training loop
# -----------------------------------------------------------------------------
results = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\nTraining {name.upper()} ...")
    pipe = Pipeline([("prep", preprocess), ("clf", model)])

    auc_scores = cross_val_score(
        pipe, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=N_JOBS,
    )
    print(f"Mean CV AUC: {auc_scores.mean():.3f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results[name] = {
        "AUC_CV_mean": auc_scores.mean(),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    # Save fitted pipeline
    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

    # SHAP explainers for tree models only
    if name in ["rf", "lgbm"]:
        print("Saving SHAP explainer ...")
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])

        X_test_pre = pipe.named_steps["prep"].transform(X_test)
        if sp.issparse(X_test_pre):
            X_test_pre = X_test_pre.toarray().astype(float)

        shap_values = explainer.shap_values(X_test_pre)

        joblib.dump(explainer, MODEL_DIR / f"{name}_explainer.joblib")
        joblib.dump(shap_values, MODEL_DIR / f"{name}_shap_values.joblib")

# -----------------------------------------------------------------------------
# 6. Persist metrics
# -----------------------------------------------------------------------------
joblib.dump(results, MODEL_DIR / "metrics.joblib")
print("\nTraining complete. Artefacts saved in:", MODEL_DIR.resolve())
