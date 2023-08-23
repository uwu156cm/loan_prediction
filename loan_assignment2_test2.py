#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess data
data = pd.read_csv("Loan.csv")
X = data[['Annual Salary', 'Bank Balance']]
y = data['Employed']


# In[6]:


# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[7]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[8]:


# Define models
models = {
    "Perceptron": Perceptron(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}


# In[9]:


# Train and compare models
best_model = None
best_accuracy = 0.0
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append((name, accuracy))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


# In[10]:


# Save best model
joblib.dump(best_model, "loan_trainModel.pkl")


# In[11]:


# Minimum bank balance requirement for loan eligibility
MIN_BANK_BALANCE = 10000
#Because the minimum deposit rate to create a bank account in most of Myanmar bank is 10000mmk


# In[12]:


# Create Streamlit app
def predict_loan_eligibility(salary, balance):
    if balance < MIN_BANK_BALANCE:
        return 0  # Not eligible due to low bank balance
    
    features = scaler.transform([[salary, balance]])
    prediction = best_model.predict(features)
    return prediction[0]

def main():
    st.title("Loan Eligibility Prediction")

    st.write("Enter the employee's annual salary and bank balance:")
    salary = st.number_input("Annual Salary", min_value=0)
    balance = st.number_input("Bank Balance", min_value=0)

    if st.button("Predict Loan Eligibility"):
        eligibility = predict_loan_eligibility(salary, balance)
        if eligibility == 1:
            st.success("Employee is eligible for a loan.")
        else:
            st.warning("Employee is not eligible for a loan.")

    st.write("## Model Comparison Results")
    df_results = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.table(df_results)


# In[13]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




