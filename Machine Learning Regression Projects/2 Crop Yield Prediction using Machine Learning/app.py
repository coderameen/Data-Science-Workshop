from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))


#create flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']
        
        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]])
        #Note, the given input is not preprocessed. but my model is train with preprocessed dataset
        #hence we need to preprocess it
        transformed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_features).reshape(1,-1)
        
        return render_template('index.html', output=predicted_value)
#python main
if __name__=='__main__':
    app.run(debug=True)