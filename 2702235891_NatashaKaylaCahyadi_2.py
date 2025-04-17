# Mid Exam Model Deployment
# Nama : Natasha Kayla Cahyadi
# NIM : 2702235891
# Kelas : LC09

import pandas as pd
import numpy as np
import pickle as pkl

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.train_encoded_home = OneHotEncoder()
        self.train_encoded_loan_intent = OneHotEncoder()
        self.rob_scaler = RobustScaler()

    def read_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df
    
    def data_preparation(self):
        print("Data Shape:")
        print(self.df.shape)
        print("\nData Info:")
        print(self.df.info())
        print("\nStatistic Descriptive:")
        print(self.df.describe())
        print("\nUnique Data:")
        print(self.df.nunique())

    def split_data(self, test_size = 0.2, random_state = 42):
        x = self.df.drop(columns=['loan_status'])
        y = self.df['loan_status']
        self.x = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state)
        
    def check_duplicated_data(self):
        print("Total Duplicated Data:")
        print(self.df.duplicated().sum())

    def check_missing_values(self):
        print("Total Missing Values:")
        print(self.df.isnull().sum())

    def handle_missing_values(self):
        print(self.x_train['person_income'].median())
        self.x_train['person_income'].fillna(67055.0, inplace=True)
        self.x_test['person_income'].fillna(67055.0, inplace=True)

    def handle_anomalies(self):
        self.x_train['person_gender'] = self.x_train['person_gender'].str.lower().str.replace(" ", "")
        self.x_train['person_gender'] = self.x_train['person_gender'].replace({
            'male': 'male',
            'female': 'female'
        })
        gender_counts = self.x_train['person_gender'].value_counts()
        print("Train Data:")
        print(gender_counts)

        self.x_test['person_gender'] = self.x_test['person_gender'].str.lower().str.replace(" ", "")
        self.x_test['person_gender'] = self.x_test['person_gender'].replace({
            'male': 'male',
            'female': 'female'
        })
        gender_counts = self.x_train['person_gender'].value_counts()
        print("\nTest Data:")
        print(gender_counts)

    def encode_binary(self):
        gender_encode = {'person_gender': {'male': 1, 'female': 0}}
        self.x_train = self.x_train.replace(gender_encode)
        self.x_test = self.x_test.replace(gender_encode)
        pkl.dump(gender_encode, open('gender_encode.pkl', 'wb'))

        previous_loan_encode = {'previous_loan_defaults_on_file': {'Yes': 1, 'No': 0}}
        self.x_train = self.x_train.replace(previous_loan_encode)
        self.x_test = self.x_test.replace(previous_loan_encode)
        pkl.dump(previous_loan_encode, open('previous_loan_defaults_on_file_encode.pkl', 'wb'))

    def encode_label(self):
        person_education_encode = {'person_education': {
            'High School': 0,
            'Associate': 1,
            'Bachelor': 2,
            'Master': 3,
            'Doctorate': 4
        }}
        self.x_train = self.x_train.replace(person_education_encode)
        self.x_test = self.x_test.replace(person_education_encode)
        pkl.dump(person_education_encode, open('person_education_encode.pkl', 'wb'))

    def encode_onehot(self):
        home_enc_train = self.x_train[['person_home_ownership']]
        loan_intent_enc_train = self.x_train[['loan_intent']]

        home_enc_test = self.x_test[['person_home_ownership']]
        loan_intent_enc_test = self.x_test[['loan_intent']]

        home_enc_train = pd.DataFrame(
            self.train_encoded_home.fit_transform(home_enc_train).toarray(),
            columns=self.train_encoded_home.get_feature_names_out()
        )
        loan_intent_enc_train = pd.DataFrame(
            self.train_encoded_loan_intent.fit_transform(loan_intent_enc_train).toarray(),
            columns=self.train_encoded_loan_intent.get_feature_names_out()
        )

        home_enc_test = pd.DataFrame(
            self.train_encoded_home.transform(home_enc_test).toarray(),
            columns=self.train_encoded_home.get_feature_names_out()
        )
        loan_intent_enc_test = pd.DataFrame(
            self.train_encoded_loan_intent.transform(loan_intent_enc_test).toarray(),
            columns=self.train_encoded_loan_intent.get_feature_names_out()
        )

        self.x_train = self.x_train.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)

        self.x_train.drop(columns=['person_home_ownership', 'loan_intent'], inplace=True)
        self.x_test.drop(columns=['person_home_ownership', 'loan_intent'], inplace=True)

        self.x_train = pd.concat([self.x_train, home_enc_train, loan_intent_enc_train], axis=1)
        self.x_test = pd.concat([self.x_test, home_enc_test, loan_intent_enc_test], axis=1)

        pkl.dump(self.train_encoded_home, open('person_home_encode.pkl', 'wb'))
        pkl.dump(self.train_encoded_loan_intent, open('loan_intent_encode.pkl', 'wb'))

    def scale_data(self):
        self.x_train = self.rob_scaler.fit_transform(self.x_train)
        self.x_test = self.rob_scaler.transform(self.x_test)

        pkl.dump(self.rob_scaler, open('robust_scaler.pkl', 'wb'))

class Modeling:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = XGBClassifier(
        n_estimators=100,          
        learning_rate=0.1,         
        subsample=0.8,             
        colsample_bytree=0.8,      
        max_depth=6,               
        random_state=42,           
        )

        self.y_pred_xgb = None
        self.y_pred_prob_xgb = None 
    
    def fit_model(self):
        self.model.fit(self.x_train, self.y_train)
    
    def make_predictions(self):
        self.y_pred_xgb = self.model.predict(self.x_test)
        self.y_pred_prob_xgb = self.model.predict_proba(self.x_test)[:, 1]

    def evaluation(self):
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_pred_xgb))
    
    def best_model(self):
        pkl.dump(self.model, open('XGB_model.pkl', 'wb'))
