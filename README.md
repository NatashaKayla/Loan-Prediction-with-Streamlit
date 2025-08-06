# 💰 Loan Prediction with Streamlit

This project builds and deploys a machine learning model to predict loan approval based on user input. It covers the full data science workflow: from exploratory data analysis and preprocessing to model training and real-time prediction using Streamlit.

---

## 🎯 Main Objectives

- 🧹 Clean and preprocess applicant loan data  
- 📊 Perform Exploratory Data Analysis (EDA) to identify patterns and outliers  
- 🧠 Train a predictive machine learning model (XGBoost)  
- 🌐 Deploy the model with a user-friendly Streamlit interface  
- ✅ Enable real-time loan approval prediction based on user inputs

---

## 🧾 Dataset Description

The dataset contains personal and financial information of loan applicants along with their loan status. Each row represents a loan application.

| Feature                          | Description                                             |
|----------------------------------|---------------------------------------------------------|
| `person_age`                     | 👤 Age of the applicant                                 |
| `person_gender`                 |  ⚥ Gender (encoded)                                     |
| `person_education`              | 🎓 Education level (encoded)                            |
| `person_income`                 | 💵 Monthly income                                       |
| `person_emp_exp`               | 🧳 Employment experience (in months)                    |
| `person_home_ownership`        | 🏠 Home ownership status (encoded)                      |
| `loan_amnt`                     | 💰 Amount of loan requested                             |
| `loan_intent`                   | 📃 Purpose of the loan (encoded)                        |
| `loan_int_rate`                 | 📈 Interest rate (×100 for percentage)                  |
| `loan_percent_income`          | 💸 Loan amount as % of monthly income                   |
| `cb_person_cred_hist_length`   | 📊 Credit history length (in years)                     |
| `credit_score`                  | 🧾 Credit score                                         |
| `previous_loan_defaults_on_file` | ❗ Past loan default record (encoded)                   |
| `loan_status`                   | ✅ Target variable: Approved or Rejected               |

---

## 🧹 Data Cleaning & Preprocessing Steps

- 🧼 Handled missing values using **median imputation**  
- 🧹 Standardized anomalies and ensured consistency in categorical entries  
- 🔁 Encoded categorical features manually using:  
  - ✅ Binary Encoding (e.g., `previous_loan_defaults_on_file`)  
  - 🏷 Label Encoding (e.g., `loan_intent`, `education`)  
- 📐 Scaled numerical features manually using **RobustScaler** to handle outliers  
- 💾 Saved the preprocessing objects (encoders and scaler) using `pickle` for deployment

---

## 📉 Exploratory Data Analysis (EDA)

The EDA aimed to understand the underlying structure of the dataset and support modeling:

- 📊 Checked **distribution of numerical features** such as income, credit score, age, and loan amount  
- 🥧 Visualized **loan status distribution** (Approved vs Rejected) using count plots  
- 📦 Identified skewed features and potential outliers

---

## 🧠 Model Development

The project uses the **XGBoost Classifier** for binary classification. It was chosen for its robustness with tabular data and ability to handle non-linearity.

Key Steps:

- 🏷 Applied manual encoding and scaling on features  
- ⚖️ Trained the model using the preprocessed data  
- ✅ Output: `"Approved"` or `"Rejected"`

---

## 🌐 Streamlit Deployment

Users can access the model via a clean and interactive Streamlit app.

### 🔧 How It Works:

1. Users fill in loan application form via the Streamlit UI  
2. Inputs are manually encoded and scaled using the saved objects  
3. The trained model makes a prediction in real-time  
4. Output: ✅ `Approved` or ❌ `Rejected`

### ▶️ Run Locally:

```bash
streamlit run 2702235891_NatashaKaylaCahyadi_3.py
````

---

## 📌 Conclusion

* 🎯 This end-to-end ML project demonstrates the full pipeline from data exploration to deployment.
* 🔍 Actionable insights from EDA informed better feature engineering and model reliability.
* ⚙️ The model accurately identifies patterns in applicant data to predict loan eligibility.
* 🖥️ Through Streamlit, the model becomes accessible and usable by non-technical users.
```
