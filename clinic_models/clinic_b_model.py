"""
Clinic B — Local Model: MLP Neural Network Classifier
Federated Learning Triage System
Data profile: Travel clinic — younger patients, high travel history
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["age", "fever", "cough", "fatigue", "travel_history", "comorbidities", "spo2"]
TARGET = "triage_level"
MODEL_PATH = "models/clinic_b_nn.pkl"
SCALER_PATH = "models/clinic_b_scaler.pkl"
ENCODER_PATH = "models/clinic_b_encoder.pkl"


def load_data():
    df = pd.read_csv("data/clinic_b_patients.csv")
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    return X, y, le


def train(rounds=5):
    """Train MLP Neural Network with RL-style hidden layer tuning."""
    X, y, le = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'='*50}")
    print("CLINIC B — MLP Neural Network Training")
    print(f"{'='*50}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    best_acc = 0
    best_model = None
    history = []

    hidden_sizes = [(32,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]

    for round_num in range(1, rounds + 1):
        arch = hidden_sizes[min(round_num - 1, len(hidden_sizes) - 1)]
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            activation="relu",
            solver="adam",
            max_iter=500,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # RL reward
        reward = "+1 (architecture expanded)" if acc > best_acc else "-1 (no gain)"
        history.append((round_num, acc))
        print(f"  Round {round_num}: accuracy = {acc:.4f} | arch = {arch} | reward = {reward}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(f"\nBest accuracy: {best_acc:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Return only model, best_acc, history (Standardizing the return tuple)
    return best_model, best_acc, history


def predict(patient_data: dict):
    """Predict triage level for a single patient."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    X = pd.DataFrame([patient_data])[FEATURES]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return {
        "clinic": "B",
        "model": "MLP Neural Network",
        "prediction": le.classes_[pred],
        "confidence": round(float(proba.max()), 4),
        "probabilities": dict(zip(le.classes_, proba.round(4).tolist()))
    }

if __name__ == "__main__":
    train(rounds=5)
