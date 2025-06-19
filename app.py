import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_ORDER = ["alcohol", "sulphates", "density", "total_sulfur_dioxide"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [float(data[key]) for key in FEATURE_ORDER]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        quality = int(prediction[0])  # 1: iyi kalite, 0: kötü kalite
        return jsonify({'quality': quality})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
