import streamlit as st
import pandas as pd
import joblib

CD4_MEDIAN = 354.5

@st.cache_resource
def load():
    return joblib.load('tpt_delay_model.pkl'), joblib.load('model_columns.pkl')

model, cols = load()

st.set_page_config(page_title="TPT Delay Risk", page_icon="🩺")
st.title("TPT Initiation Delay Risk Tool")
st.caption("Decision-support prototype. Estimates the risk that TPT will be started more than 6 months after ART.")

age = st.number_input("Age (years)", min_value=1, max_value=100, value=35)
sex = st.selectbox("Sex", ["F", "M"])
cd4_known = st.checkbox("Baseline CD4 available", value=True)
cd4 = st.number_input("Baseline CD4 (cells/mm³)", min_value=0, max_value=6000, value=350, disabled=not cd4_known)
regimen = st.selectbox("Current regimen line", ["First line", "Second line"])
ahd = st.selectbox("Advanced HIV disease (AHD) client?", ["No", "Yes"])
establishment = st.selectbox("Established in care?", ["Established", "Not Established"])
fast = st.selectbox("Fast Track differentiated care model?", ["Yes", "No"])
pmtct = st.selectbox("PMTCT status", ["No", "Pregnant", "Breastfeeding", "Unknown"])

if st.button("Estimate risk"):
    row = {c: 0 for c in cols}

    def setc(name):
        if name in row:
            row[name] = 1

    row['baseline_cd4'] = cd4 if cd4_known else CD4_MEDIAN
    row['baseline_cd4_missing'] = 0 if cd4_known else 1
    row['ahd_client'] = 1 if ahd == "Yes" else 0
    row['dsd_fast_track'] = 1 if fast == "Yes" else 0

    if sex == "M":
        setc('sex_M')

    if age <= 14: g = "<15"
    elif age <= 24: g = "15-24"
    elif age <= 34: g = "25-34"
    elif age <= 44: g = "35-44"
    elif age <= 54: g = "45-54"
    else: g = "55+"
    setc(f"age_group_{g}")

    if regimen == "Second line":
        setc("current_regimen_line_Second line")
    if establishment == "Not Established":
        setc("establishment_Not Established")
    setc(f"active_pmtct_{pmtct}")

    X = pd.DataFrame([row])[cols]
    prob = model.predict_proba(X)[0, 1]

    st.metric("Estimated risk of delayed TPT initiation", f"{prob*100:.0f}%")
    if prob >= 0.60:
        st.error("HIGH risk: prioritise TPT counselling and follow-up at the next visit.")
    elif prob >= 0.40:
        st.warning("MEDIUM risk: review TPT eligibility at the next visit.")
    else:
        st.success("LOWER risk: routine follow-up.")

st.divider()
st.caption("Proof of concept built on one facility's data (ROC-AUC about 0.70, validated by 5-fold cross-validation). "
           "It supports, and does not replace, clinical judgement.")
