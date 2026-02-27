# Fraud Risk Scoring (End-to-End ML) — Streamlit + Notebooks

This repo is made to be easy to follow if you’re new to “end-to-end ML”.

What you’ll do:
1) generate a dataset (synthetic, so it’s public-safe)
2) engineer a few fraud-style features (velocity + abnormal spend)
3) train a baseline model and save it
4) run a Streamlit app to score new transactions

---

## What’s inside
- `notebooks/01_generate_data.ipynb` → creates `data/transactions.csv`
- `notebooks/02_train_model.ipynb` → trains and saves `models/fraud_model.joblib`
- `notebooks/03_drift_check.ipynb` → quick drift check (PSI)
- `app.py` → Streamlit UI

---

## Setup
```bash
python -m venv .venv
# PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (recommended order)
1) Run `notebooks/01_generate_data.ipynb`
2) Run `notebooks/02_train_model.ipynb`
3) Start the UI:
```bash
streamlit run app.py
```

---

