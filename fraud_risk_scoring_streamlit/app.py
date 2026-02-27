import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load

st.set_page_config(page_title="Fraud Risk Scoring", layout="centered")
st.title("💳 Fraud Risk Scoring (Demo)")

artifact_path = Path("models/fraud_model.joblib")

st.sidebar.header("Model")
if artifact_path.exists():
    st.sidebar.success("Found models/fraud_model.joblib")
    artifact = load(artifact_path)
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
else:
    st.sidebar.warning("Model not found. Run notebooks/02_train_model.ipynb first.")
    artifact = None
    model = None
    threshold = 0.5

st.subheader("Enter transaction details")
c1, c2 = st.columns(2)
with c1:
    user_id = st.number_input("User ID", min_value=0, value=12)
    amount = st.number_input("Amount", min_value=1.0, value=899.0, step=1.0)
    merchant_cat = st.number_input("Merchant category (0–19)", min_value=0, max_value=19, value=3)
with c2:
    channel = st.selectbox("Channel", ["card_present", "online"], index=1)
    device_change = st.selectbox("Device change?", [0, 1], index=1)

st.caption("Try: channel=online + device_change=1 + higher amount")

def risk_band(p: float) -> str:
    if p < 0.2: return "LOW"
    if p < 0.5: return "MEDIUM"
    return "HIGH"

if st.button("Score", type="primary"):
    if model is None:
        st.error("Train the model first: notebooks/02_train_model.ipynb")
    else:
        numeric = artifact["numeric_features"]
        categorical = artifact["categorical_features"]

        # Beginner-friendly serving: use safe defaults for rolling stats.
        row = {
            "amount": float(amount),
            "device_change": int(device_change),
            "amount_z_user": 0.0,
            "txn_count_1h": 0.0,
            "txn_sum_1h": 0.0,
            "txn_max_1h": 0.0,
            "txn_count_24h": 0.0,
            "txn_sum_24h": 0.0,
            "txn_max_24h": 0.0,
            "amt_gt_3x_user_mean": 1 if amount > 500 else 0,
            "is_online": 1 if channel == "online" else 0,
            "merchant_cat": int(merchant_cat),
            "channel": channel,
        }

        X = pd.DataFrame([row], columns=numeric + categorical)
        p = float(model.predict_proba(X)[:, 1][0])

        st.metric("Fraud probability", f"{p:.3f}")
        st.metric("Risk band", risk_band(p))
        st.caption(f"Threshold saved during training: {threshold:.2f}")
