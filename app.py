import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="GenAI Burnout Risk – Model Comparison", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_PATH = os.path.join(BASE_DIR, "test_data.csv")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models…")
def load_artifacts():
    meta = joblib.load(os.path.join(MODEL_DIR, "meta.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    models = {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}
    return meta, le, models


try:
    meta, le, models = load_artifacts()
except Exception as e:
    st.error("Could not load the saved models from the `model/` folder. Check that all "
             "`.joblib` files are in the repo and that `scikit-learn` in requirements.txt "
             "matches the version used for training.")
    st.exception(e)
    st.stop()

TARGET = meta["target"]
NUM_COLS = list(meta["num_cols"])
CAT_COLS = list(meta["cat_cols"])
FEATURES = NUM_COLS + CAT_COLS
CLASS_NAMES = list(meta["class_names"])
N_CLASSES = len(CLASS_NAMES)


def train_categories():
    """Categories each categorical column had at training time (read from a saved pipeline)."""
    pipe = next(iter(models.values()))
    ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
    return dict(zip(CAT_COLS, ohe.categories_))


TRAIN_CATS = train_categories()


# --------------------------------------------------------------------------
# Data preparation and validation
# --------------------------------------------------------------------------
BOOL_MAP = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}


def prepare(df):
    """Validate the upload and return (X, y_true, warnings). Raises ValueError on fatal problems."""
    notes = []
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.dropna(subset=[TARGET]).copy()
    if df.empty:
        raise ValueError("The target column is empty.")

    unknown = sorted(set(df[TARGET].astype(str)) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unexpected target label(s) {unknown}. Expected only: {CLASS_NAMES}")

    X = df[FEATURES].copy()

    # numeric columns must really be numeric
    for c in NUM_COLS:
        coerced = pd.to_numeric(X[c], errors="coerce")
        bad = int(coerced.isna().sum() - X[c].isna().sum())
        if bad > 0:
            notes.append(f"Column `{c}` had {bad} non-numeric value(s); they were treated as missing.")
        X[c] = coerced

    # boolean-like categorical columns: make "True"/"yes"/1 match what the model saw in training
    for c in CAT_COLS:
        cats = TRAIN_CATS[c]
        if all(isinstance(v, (bool, np.bool_)) for v in cats) and X[c].dtype != bool:
            mapped = X[c].map(lambda v: BOOL_MAP.get(str(v).strip().lower(), v)
                              if pd.notna(v) else v)
            X[c] = mapped

    # unseen category values would be silently ignored by the encoder -> tell the user
    for c in CAT_COLS:
        seen = {str(v) for v in TRAIN_CATS[c]}
        new_vals = sorted({str(v) for v in X[c].dropna().unique()} - seen)
        if new_vals:
            notes.append(f"Column `{c}` has value(s) not seen in training: {new_vals}. "
                         f"Those rows are encoded as 'unknown', so predictions may be less reliable.")

    n_na = int(X.isna().sum().sum())
    if n_na:
        notes.append(f"{n_na} missing feature value(s) were filled using the training-time imputers.")

    y_true = le.transform(df[TARGET].astype(str))
    if len(df) < 30:
        notes.append("Fewer than 30 rows were uploaded; metrics will be very noisy.")
    present = set(np.unique(y_true))
    if len(present) < N_CLASSES:
        absent = [CLASS_NAMES[i] for i in range(N_CLASSES) if i not in present]
        notes.append(f"Class(es) {absent} do not appear in the uploaded data; AUC may be unavailable.")
    return X, y_true, notes


def compute_metrics(y_true, y_pred, y_proba):
    avg = "macro" if N_CLASSES > 2 else "binary"
    try:
        if N_CLASSES > 2:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                                labels=list(range(N_CLASSES)))
        else:
            auc = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner="Scoring all five models on your data…")
def score_all_models(X, y_true):
    """Run every saved model once (preprocessing applied once per model); cached per upload."""
    out = {}
    for name, pipe in models.items():
        Xt = pipe[:-1].transform(X)
        clf = pipe[-1]
        y_pred = clf.predict(Xt)
        y_proba = clf.predict_proba(Xt)
        out[name] = {"pred": y_pred, "proba": y_proba,
                     "metrics": compute_metrics(y_true, y_pred, y_proba)}
    return out


# --------------------------------------------------------------------------
# UI – sidebar
# --------------------------------------------------------------------------
st.title("GenAI Usage & Burnout Risk – Classification Models")
st.caption("Predict a student's burnout risk level (High / Low / Medium) from GenAI usage, study "
           "habits and academic features.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"],
                                help="Must contain all feature columns and the target column.")
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "rb") as f:
            st.download_button("⬇ Download sample test_data.csv", f.read(),
                               file_name="test_data.csv", mime="text/csv")
    with st.expander("Required columns"):
        st.write("**Features**")
        st.code(", ".join(FEATURES), language=None)
        st.write("**Target**")
        st.code(TARGET, language=None)

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin. You can download a ready-made sample"
            "test file from the sidebar.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the file as a CSV: {e}")
    st.stop()

try:
    X, y_true, notes = prepare(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

extra = [c for c in raw.columns if c not in FEATURES + [TARGET]]
if extra:
    notes.append(f"Ignored extra column(s): {extra}")
for n in notes:
    st.warning(n)

try:
    scored = score_all_models(X, y_true)
except Exception as e:
    st.error("The models could not score this file. The columns probably have unexpected values "
             "or types – compare with the sample test_data.csv.")
    st.exception(e)
    st.stop()

tab_sel, tab_all, tab_data = st.tabs(["Selected model", "Compare all models", "Uploaded data"])

# --------------------------------------------------------------------------
# Tab 1 – selected model
# --------------------------------------------------------------------------
with tab_sel:
    res = scored[model_name]
    y_pred, y_proba, m = res["pred"], res["proba"], res["metrics"]

    st.subheader(f"Evaluation metrics – {model_name}")
    cols = st.columns(6)
    for col, k in zip(cols, METRIC_NAMES):
        v = m[k]
        col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")
    if N_CLASSES > 2:
        st.caption("Precision, Recall and F1 are macro-averaged over the classes; AUC is one-vs-rest "
                   "macro-averaged. MCC is the multiclass Matthews correlation coefficient.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                       target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

    with st.expander("ROC curves (one-vs-rest)"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if N_CLASSES > 2:
            Yb = label_binarize(y_true, classes=list(range(N_CLASSES)))
            for i, cn in enumerate(CLASS_NAMES):
                if Yb[:, i].sum() in (0, len(Yb)):
                    continue
                fpr, tpr, _ = roc_curve(Yb[:, i], y_proba[:, i])
                ax.plot(fpr, tpr, label=f"{cn} (AUC {roc_auc_score(Yb[:, i], y_proba[:, i]):.2f})")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            ax.plot(fpr, tpr, label=f"AUC {m['AUC']:.2f}")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    pred_df = raw.loc[raw[TARGET].notna()].copy().reset_index(drop=True)
    pred_df["Predicted_" + TARGET] = le.inverse_transform(y_pred)
    st.download_button("⬇ Download predictions (CSV)", pred_df.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

# --------------------------------------------------------------------------
# Tab 2 – all models
# --------------------------------------------------------------------------
with tab_all:
    st.subheader("All five models on the uploaded test data")
    comp = pd.DataFrame({n: r["metrics"] for n, r in scored.items()}).T[METRIC_NAMES]
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#cfe8cf"))
    st.caption("Green = best model for that metric on this upload.")

    fig, ax = plt.subplots(figsize=(10, 4))
    comp.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_xticklabels(comp.index, rotation=20, ha="right")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button("⬇ Download comparison table (CSV)", comp.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# --------------------------------------------------------------------------
# Tab 3 – data overview
# --------------------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(X):,}")
    c2.metric("Features used", len(FEATURES))
    c3.metric("Classes", N_CLASSES)
    st.markdown("**Class distribution in the uploaded data**")
    dist = pd.Series(le.inverse_transform(y_true)).value_counts().reindex(CLASS_NAMES).fillna(0)
    st.bar_chart(dist)
    st.markdown("**Preview (first 10 rows)**")
    st.dataframe(raw.head(10))
