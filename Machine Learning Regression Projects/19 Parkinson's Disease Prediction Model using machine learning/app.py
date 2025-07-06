from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import pickle


#load model
model = pickle.load(open("model.pkl","rb"))
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    input_text = request.form['text']
    input_text_sep = input_text.split(",")
    np_data = np.asarray(input_text_sep, dtype=np.float32)
    prediction = model.predict(np_data.reshape(1,-1))
    print("prediction: >>>>>>>>>>>>",prediction)
    if prediction == 1:
        output = "This person has a parkinson disease"
    else:
        output = "This person doesn't have parkinson disease"
        
    return render_template("index.html", output = output)
        

if __name__=="__main__":
    app.run(debug=True)