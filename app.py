import streamlit as st
import pandas as pd
import joblib

# Page Config (white theme default)
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

# Title
st.title("💳 Credit Risk Predictor")
st.markdown("Enter the applicant's information below to predict credit risk.")

# Load the saved pipeline model
try:
    model = joblib.load("credit_pipeline.pkl")
except FileNotFoundError:
    st.error("❌ credit_pipeline.pkl file not found. Please make sure it exists in this folder.")
    st.stop()

# Input Fields
st.header("🔍 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    emp_length = st.number_input("Employment Length (Years)", min_value=0.0, value=2.0)

with col2:
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=0.2)

# Categorical Features
st.subheader("🏠 Additional Details")

home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
loan_intent = st.selectbox("Loan Intent", ['DEBTCONSOLIDATION', 'MEDICAL', 'EDUCATION', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
default_on_file = st.selectbox("Default on File", ['Y', 'N'])
cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)

# Prediction Button
if st.button("Predict Credit Risk"):
    input_data = pd.DataFrame([{
        'person_age': age,
        'person_income': income,
        'person_emp_length': emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'person_home_ownership': home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'cb_person_default_on_file': default_on_file,
        'cb_person_cred_hist_length': cred_hist_length
    }])

    # Predict using pipeline
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High Credit Risk Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Low Credit Risk (Confidence: {prob:.2f})")
