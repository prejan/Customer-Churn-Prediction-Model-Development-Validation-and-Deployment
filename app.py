import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open('churn_model_logistic.pkl', 'rb'))

st.title("📞 Customer Churn Prediction App")
st.write("This app predicts whether a telecom customer is likely to churn.")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# Convert to numeric array (simple 5 features demo)
input_data = np.array([[1 if gender=="Male" else 0, SeniorCitizen, tenure, MonthlyCharges, TotalCharges]])

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ This customer is **likely to churn!** Take retention actions.")
    else:
        st.success("✅ This customer is **not likely to churn.**")
