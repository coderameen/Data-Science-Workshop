import streamlit as st 
import numpy as np
import pandas as pd
import pickle

#load models
df = pickle.load(open('df.pkl','rb'))
rfc = pickle.load(open('rfc.pkl','rb'))



#prediction function:
def prediction(credit_score,country,gender,age,tenure,balance,products_numbers,credit_card,active_member,estimated_salary):
    input_text = (credit_score,country,gender,age,tenure,balance,products_numbers,credit_card,active_member,estimated_salary)
    np_data = np.asarray(input_text)
    prediction = rfc.predict(np_data.reshape(1,-1))
  
    if prediction == 0:
        st.write("This cusomer is still there..")
    else:
        st.write("This cusomer has left")
#WEB APP
st.title("Bank Customer Churn Prediction Application")
credit_score = st.number_input('Credit Score')
country = st.text_input('Country')
gender = st.text_input('Gender')
age = st.number_input('Age')
tenure = st.number_input('Tenure')
balance = st.number_input('Balance')
products_numbers = st.number_input('Products Number')
credit_card = st.number_input('Credit card')
active_member = st.number_input('Active Member')
estimated_salary = st.number_input('Estimated Salary')

if st.button('Predict'):
    pred = prediction(credit_score,country,gender,age,tenure,balance,products_numbers,credit_card,active_member,estimated_salary)