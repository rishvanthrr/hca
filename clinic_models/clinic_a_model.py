"""
Clinic A — Local Model: Random Forest Classifier
Federated Learning Triage System
Data profile: Elderly urban patients, high comorbidity
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["age", "fever", "cough", "fatigue", "travel_history", "comorbidities", "spo2"]
TARGET = "triage_level"
MODEL_PATH = "models/clinic_a_rf.pkl"
ENCODER_PATH = "models/clinic_a_encoder.pkl"


def load_data():
    df = pd.read_csv("data/clinic_a_patients.csv")
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    return X, y, le


def train(rounds=5):
    """Train Random Forest with RL-style iterative reward refinement."""
    X, y, le = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'='*50}")
    print("CLINIC A — Random Forest Training")
    print(f"{'='*50}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    best_acc = 0
    best_model = None
    history = []

    class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

    for round_num in range(1, rounds + 1):
        model = RandomForestClassifier(
            n_estimators=100 + round_num * 20,
            max_depth=8,
            class_weight=class_weights,
            random_state=42 + round_num
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # RL reward: boost weight of misclassified classes
        for cls in range(3):
            cls_mask = y_test == cls
            if cls_mask.sum() > 0:
                cls_acc = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
                reward = cls_acc - 0.5
                class_weights[cls] = max(0.5, class_weights[cls] - reward * 0.1)

        history.append((round_num, acc))
        print(f"  Round {round_num}: accuracy = {acc:.4f} | class_weights = { {k: round(v,3) for k,v in class_weights.items()} }")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(f"\nBest accuracy: {best_acc:.4f}")
    print("\nClassification Report:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return best_model, best_acc, history


def predict(patient_data: dict):
    """Predict triage level for a single patient."""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    X = pd.DataFrame([patient_data])[FEATURES]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "clinic": "A",
        "model": "Random Forest",
        "prediction": le.classes_[pred],
        "confidence": round(float(proba.max()), 4),
        "probabilities": dict(zip(le.classes_, proba.round(4).tolist()))
    }


if __name__ == "__main__":
    train(rounds=5)
