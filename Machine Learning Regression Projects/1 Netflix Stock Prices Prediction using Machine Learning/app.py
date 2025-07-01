import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#loading mode
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    Open  = request.form['Open']
    High  = request.form['High']
    Low  = request.form['Low']
    Adj  = request.form['Adj_Close']
    Volume  = request.form['Volume']
    year = request.form['year']
    month = request.form['month']
    day = request.form['day']
    
    features = np.array([[Open, High, Low, Adj, Volume, year, month, day]])
    features = scaler.fit_transform(features)
    prediction = model.predict(features).reshape(1,-1)
    
    return render_template('index.html', output=prediction[0])
    

if __name__=='__main__':
    app.run(debug=True)