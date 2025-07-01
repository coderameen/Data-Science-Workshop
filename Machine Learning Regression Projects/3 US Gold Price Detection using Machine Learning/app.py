#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from PIL import Image


#Load data
df = pd.read_csv('US_GOLD_PRICE_CSV.csv')
df = df.dropna()
#Split into x and y axis
X= df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

print(X.shape, " \n", y.shape)
#Split into training and testing sets
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=2)


reg = RandomForestRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
score = r2_score(y_test, pred)


#Web App
st.title("US GOLD PRICE PREDICTION MODEL USING ML")
img = Image.open('gold_img.jpg')
st.image(img,width=200,use_column_width=True)

st.subheader('Using randomforestregressor')
st.write(df)
st.subheader('Model Performance')
st.write(score)