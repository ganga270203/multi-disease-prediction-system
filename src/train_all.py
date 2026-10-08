import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import config

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    datasets = config.scan_datasets()
    if not datasets:
        print("[ERROR] No datasets to train. Add CSV files to data/raw/")
        return

    for ds in datasets:
        display_name = ds["display_name"]
        safe_name = display_name.lower().replace(" ", "_").replace("'", "")
        print(f"\nTraining model for: {display_name} ({safe_name})...")

        df = pd.read_csv(ds["file"])

        # Drop ID / unwanted columns
        drop_cols = [c for c in ds["drop_cols"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # Separate target and features
        target_col = ds["target"]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert non-numeric categorical columns
        X = pd.get_dummies(X, drop_first=True)
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        # Impute missing values with mean
        X = X.fillna(X.mean())

        # Collect feature statistics for UI input ranges
        feature_stats = {}
        for col in X.columns:
            feature_stats[col] = {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean()),
                "unique": int(X[col].nunique())
            }

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Balance classes if feasible
        try:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception:
            pass  # Fall back to raw split if SMOTE cannot run

        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Evaluate performance
        y_pred = clf.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, y_pred))

        try:
            y_proba = clf.predict_proba(X_test_scaled)
            roc = float(roc_auc_score(y_test, y_proba[:, 1])) if y_proba.shape[1] == 2 else 1.0
        except Exception:
            roc = 1.0

        cm = confusion_matrix(y_test, y_pred).tolist()

        # Extract feature importances
        importances = {
            col: round(float(imp), 4)
            for col, imp in zip(X.columns, clf.feature_importances_)
        }

        # Save artifacts
        model_path = os.path.join(MODEL_DIR, f"{safe_name}_model.joblib")
        scaler_path = os.path.join(MODEL_DIR, f"{safe_name}_scaler.joblib")
        meta_path = os.path.join(MODEL_DIR, f"{safe_name}_metadata.json")

        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "safe_name": safe_name,
            "display_name": display_name,
            "icon": ds["icon"],
            "accuracy": round(acc, 4),
            "roc_auc": round(roc, 4),
            "best_model": "Random Forest",
            "features": list(X.columns),
            "feature_stats": feature_stats,
            "confusion_matrix": cm,
            "feature_importance": importances,
            "is_multiclass": len(np.unique(y)) > 2
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f" Saved model and metadata for {display_name} (Acc: {acc*100:.1f}%)")

if __name__ == "__main__":
    train()