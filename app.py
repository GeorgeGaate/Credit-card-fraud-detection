from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('xgboost_fraud_model.pkl')
scaler = joblib.load('xgboost_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to float
        features = [float(x) for x in request.form.values()]
        # Ensure correct number of inputs
        if len(features) != 30:
            return "Error: Expected 30 features (V1–V28, Time, Amount)."
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return render_template('index.html', prediction_text=f'Transaction is: {result}')
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return f"An error occurred: {e}", 500
