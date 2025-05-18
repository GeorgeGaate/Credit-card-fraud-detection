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
        # Parse inputs
        v_features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        time = float(request.form['Time'])
        amount = float(request.form['Amount'])
        # Scale only the V features
        v_scaled = scaler.transform([v_features])
        # Combine into full input (assuming model was trained on [Time, Amount, Scaled V1-V28])
        full_input = np.hstack(([time, amount], v_scaled[0]))
        # Predict
        prediction = model.predict([full_input])[0]
        proba = model.predict_proba([full_input])[0][1] * 100
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return render_template('index.html', prediction=result, proba=f"{proba:.2f}")
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', prediction="Error during prediction.", proba="N/A")

