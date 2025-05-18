from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("xgboost_scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    prediction = int(prob > 0.5)
    return jsonify({
        "prediction": prediction,
        "probability": prob
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
