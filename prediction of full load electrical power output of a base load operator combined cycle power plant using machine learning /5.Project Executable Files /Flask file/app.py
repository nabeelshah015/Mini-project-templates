from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def index():
    return render_template("index.html")

@app.route('/data_predict', methods=['POST'])
def predict():
    at = request.form['at']
    v = request.form['v']
    ap = request.form['ap']
    rh = request.form['rh']
    
    # Prepare the input data for prediction
    data = [[float(at), float(v), float(ap), float(rh)]]
    
    # Load the model and make a prediction
    model = pickle.load(open('CCPP.pkl', 'rb'))
    prediction = model.predict(data)[0]
    
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
