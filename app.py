
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Load model artefacts ──────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load("sme_rf_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    encoders = json.load(open("encoders.json"))
    features = json.load(open("feature_cols.json"))
    return model, scaler, encoders, features

model, scaler, encoders, feature_cols = load_artefacts()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Award Predictor",
    page_icon="🏆",
    layout="centered"
)

st.title("🏆 SME Contract Award Predictor")
st.markdown(
    "Enter the details of a public procurement contract below to predict "
    "whether it is likely to be awarded to a **Small or Medium Enterprise (SME)**."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────
st.subheader("Contract Details")

col1, col2 = st.columns(2)

with col1:
    contract_value = st.number_input(
        "Contract value (£)",
        min_value=0.0, value=50000.0, step=1000.0,
        help="Estimated total value of the contract in pounds sterling"
    )
    award_year = st.selectbox(
        "Award year",
        options=list(range(2015, 2026)), index=8
    )
    award_month = st.selectbox(
        "Award month",
        options=list(range(1, 13)),
        format_func=lambda m: [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ][m-1]
    )

with col2:
    award_quarter = st.selectbox(
        "Award quarter",
        options=[1, 2, 3, 4],
        format_func=lambda q: f"Q{q}"
    )
    value_band = st.selectbox(
        "Value band",
        options=list(encoders.get("value_band", {"Unknown": 0}).keys())
    ) if "value_band" in encoders else None

    region = st.selectbox(
        "Region",
        options=list(encoders.get("region", {"Unknown": 0}).keys())
    ) if "region" in encoders else None

    cpv_code = st.selectbox(
        "CPV code",
        options=list(encoders.get("cpv_code", {"Unknown": 0}).keys())
    ) if "cpv_code" in encoders else None

st.divider()

# ── Predict ───────────────────────────────────────────────────────────
if st.button("Predict SME award probability", type="primary", use_container_width=True):

    # Build input row
    input_data = {
        "contract_value" : contract_value,
        "award_year"     : award_year,
        "award_month"    : award_month,
        "award_quarter"  : award_quarter,
    }

    # Encode categorical inputs
    for col in ["value_band", "region", "cpv_code"]:
        enc_col = col + "_enc"
        if enc_col in feature_cols:
            raw_val = locals().get(col, "Unknown") or "Unknown"
            mapping = encoders.get(col, {})
            input_data[enc_col] = mapping.get(str(raw_val), 0)

    # Build feature vector in correct order
    row = pd.DataFrame([{col: input_data.get(col, 0) for col in feature_cols}])
    row_scaled = scaler.transform(row.values)

    # Predict
    prob    = model.predict_proba(row_scaled)[0][1]
    pred    = int(prob >= 0.5)
    label   = "SME" if pred == 1 else "Non-SME"
    colour  = "green" if pred == 1 else "red"

    # Display result
    st.subheader("Prediction Result")

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Prediction",        label)
    col_r2.metric("SME Probability",   f"{prob*100:.1f}%")
    col_r3.metric("Confidence",        "High" if abs(prob-0.5) > 0.25 else "Low")

    st.progress(float(prob), text=f"SME probability: {prob*100:.1f}%")

    if pred == 1:
        st.success("✅ This contract is likely to be awarded to an SME.")
    else:
        st.warning("⚠️ This contract is unlikely to be awarded to an SME.")

    st.divider()
    st.caption(
        "Prediction made by a Random Forest classifier trained on merged "
        "Contracts Finder and Find a Tender procurement data. "
        "This is a research prototype — not for operational use."
    )

# ── Sidebar info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("About this tool")
    st.markdown(
        "This application predicts the likelihood of a UK public procurement "
        "contract being awarded to a Small or Medium Enterprise (SME).\n\n"
        "**Model:** Random Forest Classifier\n\n"
        "**Training data:** Contracts Finder + Find a Tender (merged)\n\n"
        "**Target:** sme_flag (0 = Non-SME, 1 = SME)"
    )
    st.divider()
    st.markdown("Built as part of a dissertation on SME procurement patterns.")
