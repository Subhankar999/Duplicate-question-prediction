# Duplicate Question Detector (Streamlit)

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

Files
- `app.py` – the dashboard (single prediction, CSV batch scoring, model performance)
- `features.py` – feature engineering, identical to the notebook (shared by training + app)
- `train.py` – re-train: `python train.py --csv train.csv` (writes `artifacts/`)
- `artifacts/` – trained XGBoost model, CountVectorizer, metrics
