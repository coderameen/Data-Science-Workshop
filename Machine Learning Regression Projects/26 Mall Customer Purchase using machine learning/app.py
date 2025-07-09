import streamlit as st
import pandas as pd
import pickle

#load model
model = pickle.load(open("model.pkl","rb"))

st.title("Mall Customer Purchase Prediction Application")
#st drowndown
Gender = st.selectbox('Select your Gender',(1,0))
Age = st.number_input("Enter your Age")
AIncome = st.number_input("Enter your Anual Imcome K$")

if st.button("predict"):
    prediction = model.predict([[int(Gender),int(Age),int(AIncome)]])
    st.write(prediction)