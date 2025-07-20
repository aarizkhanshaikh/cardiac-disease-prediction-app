import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def create_model_files():
    """
    This function loads the heart disease data, trains multiple ML models,
    and saves each trained model and the data scaler to separate .pkl files.
    """
    print("--- Starting: Loading and Preparing Data ---")
    
    # URL for the Heart Disease UCI dataset (processed.cleveland.data)
    try:
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None, na_values='?')
    except Exception as e:
        print(f"Failed to download data: {e}")
        return

    # Assign column names based on the dataset documentation
    df.columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # --- Data Cleaning ---
    df.dropna(inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df = df.astype(float)
    print("Data loaded and cleaned successfully.")

    # --- Feature Scaling and Data Splitting ---
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform the test data using the already fitted scaler
    X_test_scaled = scaler.transform(X_test)
    print("Data split and scaled.")

    # --- Model Training ---
    # We will train and save four different models.
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42)
    }
    
    print("\n--- Training and Saving Models ---")
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        print(f"Model '{name}' trained with Accuracy: {accuracy:.4f}")

        # --- Saving Files ---
        # This saves each model object to a unique file.
        model_filename = f'model_{name.lower()}.pkl'
        joblib.dump(model, model_filename)
        print(f"Model saved to '{model_filename}'")

    # Save the scaler object once, as it's the same for all models
    joblib.dump(scaler, 'scaler.pkl')
    print("\nScaler saved to 'scaler.pkl'")
    
    print("\n--- Process complete. You can now update app.py to use these models. ---")

if __name__ == '__main__':
    create_model_files()