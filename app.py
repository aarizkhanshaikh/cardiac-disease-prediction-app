# -*- coding: utf-8 -*-
"""
Heart Disease Prediction API (Multi-Model with MySQL Logging) - app.py
This script creates a Flask web server that serves predictions from four different
pre-trained machine learning models and logs each prediction to a MySQL database.
It also includes an endpoint to retrieve the prediction history.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import mysql.connector
from mysql.connector import Error

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- 2. Database Connection Function ---
def create_db_connection():
    """Creates and returns a connection to the MySQL database."""
    connection = None
    try:
        # IMPORTANT: Replace with your actual database credentials
        connection = mysql.connector.connect(
            host='localhost',       # Or your database server IP/hostname
            database='heart_predictions_db', # The name of your database
            user='root',            # Your database username
            password='root'
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
    return connection

# --- 3. Load All Models and the Scaler ---
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

# --- 4. Define Prediction and History API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data, makes predictions, logs them to the database,
    and returns results as a JSON object.
    """
    if not models or not scaler:
        return jsonify({'error': 'Models or scaler not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        # ... (rest of the prediction logic is the same)
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
        
        # --- Save the prediction to the database ---
        conn = create_db_connection()
        if conn:
            cursor = conn.cursor()
            sql = """INSERT INTO predictions (
                        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, 
                        oldpeak, slope, ca, thal, 
                        prediction_lr, prediction_knn, prediction_svm, prediction_rf
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            record = tuple(input_data) + (
                predictions.get('LogisticRegression'),
                predictions.get('KNN'),
                predictions.get('SVM'),
                predictions.get('RandomForest')
            )
            cursor.execute(sql, record)
            conn.commit()
            print("Prediction logged to database.")
            cursor.close()
            conn.close()

        return jsonify(predictions)

    except KeyError as e:
        return jsonify({'error': f'Missing feature in input data: {e}'}), 400
    except Error as e:
        return jsonify({'error': f'Database error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Fetches all prediction records from the database and returns them as JSON.
    """
    conn = create_db_connection()
    if conn is None:
        return jsonify({'error': 'Database connection failed.'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True) # dictionary=True returns rows as dicts
        cursor.execute("SELECT * FROM predictions ORDER BY prediction_time DESC")
        history = cursor.fetchall()
        
        # Convert datetime objects to string for JSON serialization
        for row in history:
            if 'prediction_time' in row and row['prediction_time']:
                row['prediction_time'] = row['prediction_time'].strftime('%Y-%m-%d %H:%M:%S')

        return jsonify(history)
    except Error as e:
        return jsonify({'error': f'Failed to retrieve history: {e}'}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- 5. Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)