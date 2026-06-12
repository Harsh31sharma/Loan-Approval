import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
model = pickle.load(open("Model/loan_model.pkl", "rb"))
scaler = pickle.load(open("Model/scaler.pkl", "rb"))
ohe = pickle.load(open("Model/encoder.pkl", "rb"))

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Approval Prediction System")

st.write("Enter applicant details to predict loan approval.")

# ======================
# Numerical Inputs
# ======================

Applicant_Income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=10000.0
)

Coapplicant_Income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=5000.0
)

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

Dependents = st.number_input(
    "Dependents",
    min_value=0,
    max_value=10,
    value=0
)

Credit_Score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

Existing_Loans = st.number_input(
    "Existing Loans",
    min_value=0,
    max_value=20,
    value=0
)

DTI_Ratio = st.number_input(
    "DTI Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)

Savings = st.number_input(
    "Savings",
    min_value=0.0,
    value=10000.0
)

Collateral_Value = st.number_input(
    "Collateral Value",
    min_value=0.0,
    value=20000.0
)

Loan_Amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=15000.0
)

Loan_Term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=60
)

# ======================
# Categorical Inputs
# ======================

Marital_Status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

Property_Area = st.selectbox(
    "Property Area",
    [0, 1, 2]
)

Education_Level = st.selectbox(
    "Education Level",
    [0, 1, 2]
)

Employment_Status = st.selectbox(
    "Employment Status",
    ["Contract", "Salaried", "Self-employed", "Unemployed"]
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Employer_Category = st.selectbox(
    "Employer Category",
    ["Business", "Government", "MNC", "Private", "Unemployed"]
)

# ======================
# Prediction
# ======================

if st.button("Predict Loan Status"):

    marital = 1 if Marital_Status == "Married" else 0

    cat_df = pd.DataFrame(
        [[Employment_Status,
          Loan_Purpose,
          Gender,
          Employer_Category]],
        columns=[
            "Employment_Status",
            "Loan_Purpose",
            "Gender",
            "Employer_Category"
        ]
    )

    encoded = ohe.transform(cat_df)

    numerical = np.array([
        Applicant_Income,
        Coapplicant_Income,
        Age,
        marital,
        Dependents,
        Credit_Score,
        Existing_Loans,
        DTI_Ratio,
        Savings,
        Collateral_Value,
        Loan_Amount,
        Loan_Term,
        Property_Area,
        Education_Level
    ]).reshape(1, -1)

    final_input = np.concatenate(
        [numerical, encoded],
        axis=1
    )

    final_input = scaler.transform(final_input)

    prediction = model.predict(final_input)[0]

    probability = model.predict_proba(final_input)[0][1] * 100

    if prediction == 1:
        st.success(
            f"✅ Loan Approved\n\nConfidence: {probability:.2f}%"
        )
    else:
        st.error(
            f"❌ Loan Rejected\n\nConfidence: {100-probability:.2f}%"
        )