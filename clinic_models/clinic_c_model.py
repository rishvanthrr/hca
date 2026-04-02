"""
Clinic C — Local Model: XGBoost Classifier
Federated Learning Triage System
Data profile: General community clinic — mixed population
"""

import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["age", "fever", "cough", "fatigue", "travel_history", "comorbidities", "spo2"]
TARGET = "triage_level"
MODEL_PATH = "models/clinic_c_xgb.pkl"
ENCODER_PATH = "models/clinic_c_encoder.pkl"


def load_data():
    df = pd.read_csv("data/clinic_c_patients.csv")
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    return X, y, le


def train(rounds=5):
    """Train XGBoost with RL-style parameter tuning."""
    X, y, le = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'='*50}")
    print("CLINIC C — XGBoost Training")
    print(f"{'='*50}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    best_acc = 0
    best_model = None
    history = []

    # RL initial state
    max_depth = 3
    learning_rate = 0.1

    for round_num in range(1, rounds + 1):
        model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=100 + round_num * 10,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # RL reward: tune depth and learning rate based on gain
        reward = acc - best_acc
        if reward > 0:
            max_depth = min(8, max_depth + 1)
            learning_rate = max(0.01, learning_rate * 0.9)
        else:
            max_depth = max(3, max_depth - 1)
            learning_rate = min(0.3, learning_rate * 1.1)

        history.append((round_num, acc))
        print(f"  Round {round_num}: accuracy = {acc:.4f} | depth = {max_depth} | lr = {learning_rate:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(f"\nBest accuracy: {best_acc:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Return only model, best_acc, history (Standardizing the return tuple)
    return best_model, best_acc, history


def predict(patient_data: dict):
    """Predict triage level for a single patient."""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    X = pd.DataFrame([patient_data])[FEATURES]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "clinic": "C",
        "model": "XGBoost",
        "prediction": le.classes_[pred],
        "confidence": round(float(proba.max()), 4),
        "probabilities": dict(zip(le.classes_, proba.round(4).tolist()))
    }

if __name__ == "__main__":
    train(rounds=5)
