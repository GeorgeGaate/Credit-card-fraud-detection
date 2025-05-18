from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("xgboost_scaler.pkl")

# Define the feature names in the expected order
features = [
    'Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
    'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
    'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
    'V27', 'V28'
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=features, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        scaled_input = scaler.transform([input_data])
        pred = model.predict(scaled_input)[0]
        result = "Fraud" if pred == 1 else "Not Fraud"
        return render_template("index.html", features=features, prediction=result)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
