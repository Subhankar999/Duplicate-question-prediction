import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from features import (ENGINEERED_FEATURE_NAMES, build_feature_row,
                      engineered_features, remove_punc)

st.set_page_config(page_title="Duplicate Question Detector", page_icon="🔍", layout="wide")

ART = os.path.join(os.path.dirname(__file__), "artifacts")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    booster = xgb.Booster()
    booster.load_model(os.path.join(ART, "model.json"))
    cv = joblib.load(os.path.join(ART, "vectorizer.joblib"))
    with open(os.path.join(ART, "metrics.json")) as f:
        metrics = json.load(f)
    return booster, cv, metrics


if not os.path.exists(os.path.join(ART, "model.json")):
    st.error("Model artifacts not found. Run `python train.py --csv train.csv` first.")
    st.stop()

booster, cv, metrics = load_artifacts()


def predict_proba(pairs):
    """pairs: list[(q1, q2)] -> np.array of P(duplicate)."""
    out = []
    for i in range(0, len(pairs), 2000):  # chunk to keep memory low
        X = np.vstack([build_feature_row(a, b, cv) for a, b in pairs[i:i + 2000]])
        out.append(booster.predict(xgb.DMatrix(X)))
    return np.concatenate(out) if out else np.array([])


def gauge(p, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Duplicate probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2e7d32" if p >= threshold else "#c62828"},
            "steps": [{"range": [0, threshold * 100], "color": "rgba(198,40,40,0.12)"},
                      {"range": [threshold * 100, 100], "color": "rgba(46,125,50,0.12)"}],
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
        }))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🔍 Duplicate Detector")
    st.caption("XGBoost on engineered text-similarity features + bag-of-words, "
               "trained on the Quora Question Pairs dataset.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Probability at or above this value is labelled *duplicate*. "
                               "Lower it to catch more duplicates (more false alarms); "
                               "raise it to be stricter.")
    st.divider()
    st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.caption(f"Trained on {metrics['n_train']:,} pairs, tested on {metrics['n_test']:,}.")

tab_predict, tab_batch, tab_perf = st.tabs(["🔎 Single prediction", "📄 Batch (CSV)", "📊 Model performance"])

# --------------------------------------------------------------------------- #
# Tab 1 – single prediction
# --------------------------------------------------------------------------- #
with tab_predict:
    st.header("Are these two questions duplicates?")

    examples = {"— write your own —": ("", "")}
    for e in metrics.get("examples", []):
        tag = "✅ dup" if e["label"] else "❌ not dup"
        examples[f"[{tag}] {e['q1'][:55]}…"] = (e["q1"], e["q2"])
    examples["[✅ dup] How can I learn Python? / What is the best way to learn Python?"] = (
        "How can I learn Python?", "What is the best way to learn Python?")
    examples["[❌ not dup] How can I lose weight? / How can I gain weight?"] = (
        "How can I lose weight?", "How can I gain weight?")

    choice = st.selectbox("Try an example", list(examples.keys()))
    ex_q1, ex_q2 = examples[choice]

    # keyed by the chosen example so the boxes refresh when it changes
    c1, c2 = st.columns(2)
    q1 = c1.text_area("Question 1", value=ex_q1, height=110, key=f"q1_{choice}",
                      placeholder="e.g. What is the capital of India?")
    q2 = c2.text_area("Question 2", value=ex_q2, height=110, key=f"q2_{choice}",
                      placeholder="e.g. Which city is the capital of India?")

    if st.button("Predict", type="primary", width="stretch"):
        if not q1.strip() or not q2.strip():
            st.warning("Please enter both questions.")
        else:
            p = float(predict_proba([(q1, q2)])[0])
            is_dup = p >= threshold

            left, right = st.columns([1, 1])
            with left:
                if is_dup:
                    st.success("### ✅ Duplicate\nThese questions most likely ask the same thing.")
                else:
                    st.error("### ❌ Not duplicate\nThese questions most likely ask different things.")
                st.plotly_chart(gauge(p, threshold), width="stretch")
            with right:
                st.subheader("Why? Similarity signals")
                f = dict(zip(ENGINEERED_FEATURE_NAMES, engineered_features(q1, q2)))
                w1 = {w.lower() for w in remove_punc(q1).split()}
                w2 = {w.lower() for w in remove_punc(q2).split()}
                shared = sorted(w1 & w2)
                st.markdown("**Shared words:** " + (", ".join(f"`{w}`" for w in shared) or "_none_"))
                only1 = sorted(w1 - w2)
                only2 = sorted(w2 - w1)
                st.markdown("**Only in Q1:** " + (", ".join(f"`{w}`" for w in only1) or "_none_"))
                st.markdown("**Only in Q2:** " + (", ".join(f"`{w}`" for w in only2) or "_none_"))

                sig = pd.DataFrame({
                    "Signal": ["Fuzzy ratio", "Partial ratio", "Token-sort ratio", "Token-set ratio",
                               "Word share", "Longest common substring"],
                    "Score": [f["fuzz_ratio"] / 100, f["fuzz_partial_ratio"] / 100,
                              f["tokensort_ratio"] / 100, f["tokenset_ratio"] / 100,
                              f["word_share"], f["longest_substr"]],
                })
                st.dataframe(sig, hide_index=True, width="stretch",
                             column_config={"Score": st.column_config.ProgressColumn(
                                 "Score", min_value=0, max_value=1, format="%.2f")})

            with st.expander("All engineered features"):
                st.dataframe(pd.DataFrame({"feature": list(f.keys()), "value": list(f.values())}),
                             hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2 – batch
# --------------------------------------------------------------------------- #
with tab_batch:
    st.header("Score many pairs at once")
    st.write("Upload a CSV with columns **question1** and **question2**. "
             "If it also has an **is_duplicate** column, accuracy is reported.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = {"question1", "question2"} - set(data.columns)
        if missing:
            st.error(f"Missing column(s): {', '.join(sorted(missing))}")
        else:
            data = data.dropna(subset=["question1", "question2"]).reset_index(drop=True)
            cap = 20000
            if len(data) > cap:
                st.warning(f"Only the first {cap:,} rows will be scored.")
                data = data.head(cap)
            with st.spinner(f"Scoring {len(data):,} pairs…"):
                data["duplicate_probability"] = predict_proba(
                    list(zip(data.question1.astype(str), data.question2.astype(str)))).round(4)
            data["prediction"] = (data.duplicate_probability >= threshold).astype(int)

            m1, m2, m3 = st.columns(3)
            m1.metric("Pairs scored", f"{len(data):,}")
            m2.metric("Predicted duplicates", f"{int(data.prediction.sum()):,}")
            if "is_duplicate" in data.columns:
                m3.metric("Accuracy vs is_duplicate",
                          f"{(data.prediction == data.is_duplicate).mean():.1%}")
            st.dataframe(data, width="stretch", hide_index=True)
            st.download_button("⬇️ Download predictions", data.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Tab 3 – performance
# --------------------------------------------------------------------------- #
with tab_perf:
    st.header("Model performance (held-out 20 % test set)")
    rep = metrics["report"]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c.metric("Precision (dup)", f"{rep['Duplicate']['precision']:.1%}")
    d.metric("Recall (dup)", f"{rep['Duplicate']['recall']:.1%}")

    l, r = st.columns(2)
    with l:
        cm = np.array(metrics["confusion_matrix"])
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Pred: not dup", "Pred: dup"], y=["True: not dup", "True: dup"],
            text=cm, texttemplate="%{text:,}", colorscale="Blues", showscale=False))
        fig.update_layout(title="Confusion matrix", height=380, yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with r:
        imp = pd.DataFrame(metrics["importance"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.gain, y=imp.feature, orientation="h"))
        fig.update_layout(title="Top 20 features (gain)", height=380, margin=dict(l=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")
    st.caption(f"Dataset duplicate rate: {metrics['dup_rate']:.1%} "
               "(about 37 % of Quora pairs are duplicates, so the classes are imbalanced).")
