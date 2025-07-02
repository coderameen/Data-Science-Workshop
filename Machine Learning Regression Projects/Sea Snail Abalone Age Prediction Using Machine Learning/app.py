from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle


#Loading Model
model = pickle.load(open('model.pkl','rb'))

#Create flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    sex = request.form['sex']
    len = request.form['len']
    dim = request.form['dim']
    height = request.form['ht']
    whw = request.form['whw']
    sw = request.form['sw']
    vw = request.form['vw']
    shw = request.form['shw']
    
    
    features = np.array([[sex,len,dim,height,whw,sw,vw,shw]])
    prediction = model.predict(features).reshape(1,-1)
    return render_template("index.html", age=prediction[0])
#Python Main
if __name__=='__main__':
    app.run(debug=True)