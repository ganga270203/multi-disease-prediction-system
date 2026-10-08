import os
import glob
import pandas as pd

RAW_DIR = "data/raw"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Known dataset configurations (auto-matched by filename keywords)
KNOWN_CONFIGS = {
    "diabetes": {
        "display_name": "Diabetes",
        "target": "Outcome",
        "drop_cols": [],
        "icon": "🩸"
    },
    "heart": {
        "display_name": "Heart Disease",
        "target": "target",
        "drop_cols": [],
        "icon": "❤️"
    },
    "symptom": {
        "display_name": "Patient Profile & Symptoms",
        "target": "Outcome Variable",
        "drop_cols": ["Disease"],
        "icon": "📋"
    },
    "disease_dataset": {
        "display_name": "General Disease Predictor",
        "target": "disease",
        "drop_cols": [],
        "icon": "🦠"
    },
    "parkinson": {
        "display_name": "Parkinson's Disease",
        "target": "status",
        "drop_cols": ["name"],
        "icon": "🧠"
    },
    "liver": {
        "display_name": "Liver Disease",
        "target": "Dataset",
        "drop_cols": [],
        "icon": "🫀"
    },
    "kidney": {
        "display_name": "Kidney Disease",
        "target": "classification",
        "drop_cols": ["id"],
        "icon": "🫁"
    }
}

TARGET_KEYWORDS = ["outcome", "target", "status", "dataset",
                   "classification", "result", "diagnosis", "disease"]
ID_KEYWORDS = ["id", "name", "patient"]


def detect_target(df, filename=""):
    """Try to match known config, then auto-detect."""
    fname_lower = filename.lower()
    for key, cfg in KNOWN_CONFIGS.items():
        if key in fname_lower:
            if cfg["target"] in df.columns:
                return cfg["target"], cfg.get("drop_cols", [])

    # Fallback auto-detect
    for col in df.columns:
        if col.lower() in TARGET_KEYWORDS:
            return col, []

    raise ValueError(
        f"Cannot detect target in '{filename}'. "
        "Rename the target column to 'target' or 'Outcome'."
    )


def get_display_name(filename):
    fname_lower = filename.lower()
    for key, cfg in KNOWN_CONFIGS.items():
        if key in fname_lower:
            return cfg["display_name"], cfg.get("icon", "🏥")
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.replace("_", " ").title(), "🏥"


def scan_datasets():
    """Scan data/raw/ and return list of dataset configs."""
    files = glob.glob(f"{RAW_DIR}/*.csv")
    if not files:
        print(f"[WARN] No CSV files found in {RAW_DIR}/")
        return []

    datasets = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=5)
            target, drop_cols = detect_target(df, f)
            display_name, icon = get_display_name(f)
            full_df = pd.read_csv(f)
            datasets.append({
                "file": f,
                "display_name": display_name,
                "icon": icon,
                "target": target,
                "drop_cols": drop_cols,
                "rows": len(full_df),
                "cols": len(full_df.columns)
            })
            print(f"[CONFIG] {display_name}: target='{target}', rows={len(full_df)}")
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    return datasets


if __name__ == "__main__":
    datasets = scan_datasets()
    print(f"\nDetected {len(datasets)} dataset(s): {[d['display_name'] for d in datasets]}")