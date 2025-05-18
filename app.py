from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('xgboost_fraud_model.pkl')
scaler = joblib.load('xgboost_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        input_features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        time = float(request.form['Time'])
        amount = float(request.form['Amount'])
        # Combine features into a single array
        features = [time, amount] + input_features
        input_array = np.array([features])
        # Scale input
        input_scaled = scaler.transform(input_array)
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1] * 100
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return render_template('index.html', prediction=result, proba=f"{proba:.2f}")
    except Exception as e:
        # Log the error
        print(f"Prediction error: {e}")
        return render_template('index.html', prediction="Error during prediction.", proba="N/A")
