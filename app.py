# -*- coding: utf-8 -*-
"""
Heart Disease Prediction API (Stateless Multi-Model) - app.py
This script creates a Flask web server that serves predictions from four different
pre-trained machine learning models. It is stateless and does not log data.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- 2. Load All Models and the Scaler ---
models = {}
model_names = ['LogisticRegression', 'KNN', 'SVM', 'RandomForest']

try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully.")
    for name in model_names:
        filename = f'model_{name.lower()}.pkl'
        models[name] = joblib.load(filename)
        print(f"Model '{filename}' loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: A required .pkl file was not found. {e}")
    models = {}
except Exception as e:
    print(f"An error occurred while loading files: {e}")
    models = {}

# --- 3. Define Prediction API Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data, makes predictions, and returns results as JSON.
    """
    if not models or not scaler:
        return jsonify({'error': 'Models or scaler not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        input_data = [data[feature] for feature in feature_names]
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        predictions = {}
        for name, model in models.items():
            prediction = model.predict(scaled_input)
            predictions[name] = int(prediction[0])
        
        return jsonify(predictions)

    except KeyError as e:
        return jsonify({'error': f'Missing feature in input data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

# --- 4. Run the Flask App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)
