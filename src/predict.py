import os
import json
import glob
import joblib
import pandas as pd

MODEL_DIR = "models"

def list_trained_diseases():
    """Reads all metadata JSON files from the models directory."""
    meta_files = glob.glob(f"{MODEL_DIR}/*_metadata.json")
    diseases = []
    for mf in meta_files:
        try:
            with open(mf, "r") as f:
                data = json.load(f)
                diseases.append(data)
        except Exception:
            continue
    return diseases

def load_model(safe_name):
    """Loads the model artifact, scaler, and metadata JSON."""
    model_path = os.path.join(MODEL_DIR, f"{safe_name}_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{safe_name}_scaler.joblib")
    meta_path = os.path.join(MODEL_DIR, f"{safe_name}_metadata.json")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, scaler, meta

def predict_single(safe_name, input_dict):
    """Runs single-patient inference and produces risk classifications."""
    model, scaler, meta = load_model(safe_name)
    features = meta["features"]

    # Align inputs to training feature columns
    df_input = pd.DataFrame([[input_dict.get(f, 0.0) for f in features]], columns=features)
    X_scaled = scaler.transform(df_input)

    pred = int(model.predict(X_scaled)[0])
    
    # Calculate probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        prob = float(probs[1] if len(probs) == 2 else probs[pred]) * 100
    else:
        prob = 100.0 if pred == 1 else 0.0

    # Risk level classification
    if prob >= 65:
        risk_level = "High"
        severity = "High"
        message = "Clinical markers suggest an elevated risk. Consult a physician for validation."
    elif prob >= 35:
        risk_level = "Moderate"
        severity = "Medium"
        message = "Borderline values detected. Routine screening and lifestyle changes are recommended."
    else:
        risk_level = "Low"
        severity = "Low"
        message = "Indicators remain within healthy thresholds. Continue preventive health monitoring."

    recs = [
        "Schedule periodic health examinations and routine blood tests.",
        "Maintain a balanced, nutrient-dense diet and stay hydrated.",
        "Engage in regular physical exercise appropriate for your age."
    ]

    return {
        "prediction": pred,
        "prediction_label": f"Class {pred}",
        "probability": round(prob, 1),
        "risk_level": risk_level,
        "severity": severity,
        "message": message,
        "recommendations": recs,
        "top_classes": [{"label": f"Class {pred}", "prob": round(prob, 1)}]
    }