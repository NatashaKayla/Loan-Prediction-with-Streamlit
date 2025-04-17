# Mid Exam Model Deployment
# Nama : Natasha Kayla Cahyadi
# NIM : 2702235891
# Kelas : LC09

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model =  joblib.load("XGB_model.pkl")
gender_encode = joblib.load('gender_encode.pkl')
loan_intent_encode = joblib.load('loan_intent_encode.pkl')
person_education_encode = joblib.load('person_education_encode.pkl')
person_home_encode = joblib.load('person_home_encode.pkl')
previous_loan_encode = joblib.load('previous_loan_defaults_on_file_encode.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')

class LoanStatusApp:
    def __init__(self):
        self.load_models()

    def load_models(self):
        self.model = joblib.load('XGB_model.pkl')
        self.gender_encode = joblib.load('gender_encode.pkl')
        self.loan_intent_encode = joblib.load('loan_intent_encode.pkl')
        self.person_education_encode = joblib.load('person_education_encode.pkl')
        self.person_home_encode = joblib.load('person_home_encode.pkl')
        self.previous_loan_encode = joblib.load('previous_loan_defaults_on_file_encode.pkl')
        self.robust_scaler = joblib.load('robust_scaler.pkl')

    def user_input(self):
        st.title('Loan Status Model Deployment')
        # Add user input components
        person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
        person_gender = st.radio("Gender", ["male", "female"])
        person_education = st.selectbox("Highest Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_income = st.number_input("Annual Income", min_value=0)
        person_emp_exp = st.number_input("Years of Work Experience", min_value=0, max_value=100, step=1)
        person_home_ownership = st.selectbox("Home Ownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_amnt = st.number_input("Requested Loan Amount", min_value=0)
        loan_intent = st.selectbox("Purpose of Loan", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.01)
        loan_percent_income = st.number_input("Loan Amount as % of Income", min_value=0.0, max_value=100.0, step=0.01)
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=100)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
        previous_loan_defaults_on_file = st.radio("Previous Loan Defaults", ["Yes", "No"])

        # Store user input into a dictionary
        user_input = {
        'person_age': int(person_age),
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': int(person_income),
        'person_emp_exp': int(person_emp_exp),
        'person_home_ownership': person_home_ownership,
        'loan_amnt': int(loan_amnt),
        'loan_intent': loan_intent,
        'loan_int_rate': float(loan_int_rate),
        'loan_percent_income': float(loan_percent_income),
        'cb_person_cred_hist_length': int(cb_person_cred_hist_length),
        'credit_score': int(credit_score),
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }
        
        return pd.DataFrame([user_input])

    def preprocess_data(self, df):
        # Label Encoding
        df.replace(self.gender_encode, inplace=True)
        df.replace(self.previous_loan_encode, inplace=True)
        df.replace(self.person_education_encode, inplace=True)

        # One Hot Encoding
        cat_intent=df[['loan_intent']]
        cat_home=df[['person_home_ownership']]
        cat_enc_intent=pd.DataFrame(loan_intent_encode.transform(cat_intent).toarray(),columns=loan_intent_encode.get_feature_names_out())
        cat_enc_home=pd.DataFrame(person_home_encode.transform(cat_home).toarray(),columns=person_home_encode.get_feature_names_out())
        df=pd.concat([df,cat_enc_intent,cat_enc_home], axis=1)
        df=df.drop(['loan_intent', 'person_home_ownership'],axis=1)
        
        expected_columns = [
            'person_age', 'person_gender', 'person_education', 'person_income', 'person_emp_exp', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file',
            'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
            'person_home_ownership_OWN', 'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
            'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
            'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        ]
        
        # Reorder columns to exactly match training data
        df = df[expected_columns]

        # Scaling data
        scaled_df = self.robust_scaler.transform(df)
        return scaled_df

    def prediction(self, input):
        result = self.model.predict(input)
        return "Approved" if result[0] == 1 else "Rejected"
    
    def run(self):
        user_df = self.user_input()
        if st.button("Predict Loan Status"):
            processed_input = self.preprocess_data(user_df)
            prediction = self.prediction(processed_input)
            st.success(f"Loan Status: {prediction}")

if __name__ == "__main__":
    app = LoanStatusApp()
    app.run()
