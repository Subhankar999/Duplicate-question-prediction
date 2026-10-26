"""
Train the duplicate-question model exactly as in the notebook (d1.ipynb)
and save everything the Streamlit app needs into ./artifacts.

Usage:  python train.py --csv train.csv
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

from features import ENGINEERED_FEATURE_NAMES, engineered_features, remove_punc

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="train.csv")
ap.add_argument("--sample", type=int, default=70000)
ap.add_argument("--out", default="artifacts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

t0 = time.time()
df = pd.read_csv(args.csv).dropna()
df = df.sample(args.sample, random_state=42).reset_index(drop=True)
print(f"rows: {len(df)}  duplicate rate: {df.is_duplicate.mean():.3f}", flush=True)

# ---- engineered features (raw text) ----
eng = np.array([engineered_features(a, b) for a, b in zip(df.question1, df.question2)],
               dtype=np.float32)
print(f"engineered features done  {time.time()-t0:.0f}s", flush=True)

# ---- bag of words (punctuation removed), fitted on q1 + q2 ----
c1 = df.question1.map(remove_punc)
c2 = df.question2.map(remove_punc)
cv = CountVectorizer(max_features=3000, ngram_range=(1, 2))
cv.fit(list(c1) + list(c2))
b1 = cv.transform(c1).astype(np.float32)
b2 = cv.transform(c2).astype(np.float32)
print(f"BoW done  {time.time()-t0:.0f}s", flush=True)

y = df.is_duplicate.values
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)


class Chunks(xgb.DataIter):
    """Feed dense float32 chunks so we never hold the full dense matrix in RAM."""
    def __init__(self, idx, size=5000):
        super().__init__()
        self.idx, self.size, self.i = idx, size, 0

    def next(self, input_data):
        if self.i >= len(self.idx):
            return 0
        sel = self.idx[self.i:self.i + self.size]
        X = np.hstack((eng[sel], b1[sel].toarray(), b2[sel].toarray()))
        input_data(data=X, label=y[sel])
        self.i += self.size
        return 1

    def reset(self):
        self.i = 0


dtrain = xgb.QuantileDMatrix(Chunks(idx_tr), max_bin=256)
dtest = xgb.QuantileDMatrix(Chunks(idx_te), ref=dtrain, max_bin=256)

# same defaults as XGBClassifier() in the notebook
params = dict(objective="binary:logistic", eval_metric="logloss", max_depth=6,
              eta=0.3, tree_method="hist", nthread=os.cpu_count(), seed=42)
booster = xgb.train(params, dtrain, num_boost_round=100,
                    evals=[(dtrain, "train"), (dtest, "test")], verbose_eval=20)
print(f"training done  {time.time()-t0:.0f}s", flush=True)

proba = booster.predict(dtest)
pred = (proba >= 0.5).astype(int)
y_te = y[idx_te]
metrics = {
    "accuracy": float(accuracy_score(y_te, pred)),
    "roc_auc": float(roc_auc_score(y_te, proba)),
    "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    "report": classification_report(y_te, pred, output_dict=True,
                                    target_names=["Not duplicate", "Duplicate"]),
    "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
    "dup_rate": float(y.mean()),
}
print(json.dumps({k: metrics[k] for k in ("accuracy", "roc_auc")}), flush=True)

# ---- feature importance (gain) with readable names ----
names = (ENGINEERED_FEATURE_NAMES
         + [f"q1 word: {w}" for w in cv.get_feature_names_out()]
         + [f"q2 word: {w}" for w in cv.get_feature_names_out()])
gain = booster.get_score(importance_type="gain")
imp = sorted(((names[int(k[1:])], v) for k, v in gain.items()), key=lambda t: -t[1])[:20]
metrics["importance"] = [{"feature": n, "gain": float(v)} for n, v in imp]

# ---- a few real held-out examples for the app's "try an example" menu ----
te = df.iloc[idx_te].copy()
te["proba"] = proba
ex = pd.concat([te[te.is_duplicate == 1].head(6), te[te.is_duplicate == 0].head(6)])
metrics["examples"] = [
    {"q1": r.question1, "q2": r.question2, "label": int(r.is_duplicate)}
    for r in ex.itertuples()
]

booster.save_model(os.path.join(args.out, "model.json"))
joblib.dump(cv, os.path.join(args.out, "vectorizer.joblib"))
json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
print(f"saved to {args.out}/   total {time.time()-t0:.0f}s")
