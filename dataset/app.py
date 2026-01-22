import streamlit as st
import numpy as np
import joblib

# Load pre-trained models and scaler
model_linear = joblib.load("model_linear.pkl")
model_poly   = joblib.load("model_poly.pkl")
model_rbf    = joblib.load("model_rbf.pkl")
scaler       = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

st.sidebar.header("Applicant Details")
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Employed", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

kernel_choice = st.sidebar.radio("Select SVM Kernel", ["Linear", "Polynomial", "RBF"])

# Encode inputs
credit_history_num = 1 if credit_history == "Yes" else 0
employment_num = 1 if employment == "Employed" else 0
property_area_num = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

user_input = np.array([[applicant_income, coapplicant_income, loan_amount,
                        credit_history_num, employment_num, property_area_num]])

user_scaled = scaler.transform(user_input)

if st.button("Check Loan Eligibility"):
    if kernel_choice == "Linear":
        model = model_linear
    elif kernel_choice == "Polynomial":
        model = model_poly
    else:
        model = model_rbf

    prediction = model.predict(user_scaled)[0]
    confidence = model.predict_proba(user_scaled).max()

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {confidence*100:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected! (Confidence: {confidence*100:.2f}%)")

    st.info(f"Kernel used: {kernel_choice}")

    if prediction == 1:
        st.write("Based on credit history and income pattern, the applicant is likely to repay the loan.")
    else:
        st.write("Based on credit history and income pattern, the applicant is unlikely to repay the loan.")
