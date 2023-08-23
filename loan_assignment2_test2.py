#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
import streamlit as st

# Load the dataset
data = pd.read_csv('Loan.csv')


# In[5]:


# Preprocessing
X = data[['Employed', 'Annual Salary']].values
y = data['Bank Balance'].apply(lambda x: 1 if x > 0 else 0).values


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Initialize models
models = [
    ('Perceptron', Perceptron()),
    ('Logistic Regression', LogisticRegression()),
    ('SVM', SVC())
]


# In[8]:


# Model comparison
best_model = None
best_accuracy = 0

for name, model in models:
    # Cross-validation to estimate model performance
    scores = cross_val_score(model, X_train, y_train, cv=5)
    avg_accuracy = np.mean(scores)
    
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_model = model


# In[9]:


# Train the best model on the full training set
best_model.fit(X_train, y_train)


# In[11]:


# Streamlit app
st.title('Loan Eligibility Prediction')

salary = st.number_input('Enter your annual salary:')

if st.button('Predict'):
    employed = 1  # For simplicity, assuming the user is employed
    input_data = np.array([[employed, salary]])
    prediction = best_model.predict(input_data)
    
    if prediction == 1:
        st.write('Congratulations! You are likely eligible for a loan.')
    else:
        st.write('Sorry, you might not be eligible for a loan.')


# In[ ]:




