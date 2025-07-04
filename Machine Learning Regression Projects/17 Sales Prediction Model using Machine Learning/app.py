from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to preprocess the input data
def preprocess_input(input_str):
    input_list = input_str.split(',')
    input_array = np.array(input_list, dtype=np.float32)
    return input_array.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_str = request.form['input_string']
    input_array = preprocess_input(input_str)
    prediction = model.predict(input_array)[0]
    output_message = f'The predicted sales value is {prediction:.2f}'
    return render_template('index.html', prediction_text=output_message)

if __name__ == '__main__':
    app.run(debug=True)
