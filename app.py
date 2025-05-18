from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model_path = 'xgboost_fraud_model.pkl'
scaler_path = 'xgboost_scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# List of features used in the model (you can also auto-load this if you saved it)
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', feature_names=feature_names, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form.get(col)) for col in feature_names]
        scaled_data = scaler.transform([data])
        prediction = int(model.predict(scaled_data)[0])
        return render_template('index.html', feature_names=feature_names, prediction=prediction)
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

# For API use
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        input_data = request.json
        data = [input_data[col] for col in feature_names]
        scaled_data = scaler.transform([data])
        prediction = int(model.predict(scaled_data)[0])
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
