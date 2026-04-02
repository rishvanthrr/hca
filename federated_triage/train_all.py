import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinic_models.clinic_a_model import train as train_a
from clinic_models.clinic_b_model import train as train_b
from clinic_models.clinic_c_model import train as train_c
import json

def train_all(rounds=5):
    print("\n" + "="*60)
    print("  FEDERATED TRIAGE — LOCAL MODEL TRAINING")
    print("="*60)

    results = {}

    model_a, acc_a, hist_a = train_a(rounds=rounds)
    results["A"] = {"model": "Random Forest", "accuracy": round(acc_a, 4), "history": hist_a}

    model_b, acc_b, hist_b = train_b(rounds=rounds)
    results["B"] = {"model": "MLP Neural Network", "accuracy": round(acc_b, 4), "history": hist_b}

    model_c, acc_c, hist_c = train_c(rounds=rounds)
    results["C"] = {"model": "XGBoost", "accuracy": round(acc_c, 4), "history": hist_c}

    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    for clinic, res in results.items():
        print(f"  Clinic {clinic} ({res['model']}): {res['accuracy']*100:.2f}% accuracy")

    avg_acc = sum(r["accuracy"] for r in results.values()) / 3
    print(f"\n  Average local accuracy: {avg_acc*100:.2f}%")
    print("="*60)

    os.makedirs("models", exist_ok=True)
    with open("models/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to models/training_results.json")

    return results

if __name__ == "__main__":
    train_all(rounds=5)
