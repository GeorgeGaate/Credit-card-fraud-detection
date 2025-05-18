from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('xgboost_fraud_model.pkl')
scaler = joblib.load('xgboost_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        input_features.insert(0, float(request.form['Amount']))
        input_features.insert(0, float(request.form['Time']))
        input_array = np.array([input_features])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1] * 100
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return render_template('index.html', prediction=result, proba=f"{proba:.2f}")
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
