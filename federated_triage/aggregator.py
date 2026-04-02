import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinic_models.clinic_a_model import predict as predict_a
from clinic_models.clinic_b_model import predict as predict_b
from clinic_models.clinic_c_model import predict as predict_c
import json

def get_model_weights():
    try:
        with open("models/training_results.json", "r") as f:
            res = json.load(f)
            acc_a = res["A"]["accuracy"]
            acc_b = res["B"]["accuracy"]
            acc_c = res["C"]["accuracy"]
            total = acc_a + acc_b + acc_c
            return acc_a/total, acc_b/total, acc_c/total
    except Exception:
        return 0.33, 0.33, 0.34

def aggregate_predictions(patient_data):
    """Perform soft voting ensemble using accuracy-based weights."""
    pred_a = predict_a(patient_data)
    pred_b = predict_b(patient_data)
    pred_c = predict_c(patient_data)
    
    w_a, w_b, w_c = get_model_weights()
    
    classes = ["Low", "Medium", "High"]
    global_probs = {c: 0.0 for c in classes}
    
    for c in classes:
        global_probs[c] += pred_a["probabilities"].get(c, 0) * w_a
        global_probs[c] += pred_b["probabilities"].get(c, 0) * w_b
        global_probs[c] += pred_c["probabilities"].get(c, 0) * w_c
        
    final_pred = max(global_probs, key=global_probs.get)
    
    return {
        "final_prediction": final_pred,
        "global_probabilities": {k: round(v, 4) for k, v in global_probs.items()},
        "clinic_a": pred_a,
        "clinic_b": pred_b,
        "clinic_c": pred_c,
        "weights": {"A": round(w_a, 4), "B": round(w_b, 4), "C": round(w_c, 4)}
    }
