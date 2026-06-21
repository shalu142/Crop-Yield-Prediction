# train_model.py
import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# =============================
# Load Data
# =============================
df1 = pd.read_csv("Crops_data_ultra_cleaned.csv")

# =============================
# Define crops and their columns
# =============================
crop_columns = {
    "Rice": {
        "area": "RICE AREA (1000 ha)",
        "production": "RICE PRODUCTION (1000 tons)",
        "yield": "RICE YIELD (Kg per ha)"
    },
    "Wheat": {
        "area": "WHEAT AREA (1000 ha)",
        "production": "WHEAT PRODUCTION (1000 tons)",
        "yield": "WHEAT YIELD (Kg per ha)"
    },
    "Chickpea": {
        "area": "CHICKPEA AREA (1000 ha)",
        "production": "CHICKPEA PRODUCTION (1000 tons)",
        "yield": "CHICKPEA YIELD (Kg per ha)"
    },
    "Maize": {
        "area": "MAIZE AREA (1000 ha)",
        "production": "MAIZE PRODUCTION (1000 tons)",
        "yield": "MAIZE YIELD (Kg per ha)"
    },
    "Cotton": {
        "area": "COTTON AREA (1000 ha)",
        "production": "COTTON PRODUCTION (1000 tons)",
        "yield": "COTTON YIELD (Kg per ha)"
    },
    "Sugarcane": {
        "area": "SUGARCANE AREA (1000 ha)",
        "production": "SUGARCANE PRODUCTION (1000 tons)",
        "yield": "SUGARCANE YIELD (Kg per ha)"
    }
}

# =============================
# Model Training
# =============================
rf_params = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [5, 10, 15, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", None]
}

crop_stats = {}

for crop, cols in crop_columns.items():
    print(f"\n🌾 Training model for {crop}...")

    # Build dataset for this crop
    if not all(c in df1.columns for c in cols.values()):
        print(f"⚠️ Skipping {crop}, missing columns in dataset.")
        continue

    crop_df = df1[[cols["area"], cols["production"], cols["yield"]]].dropna()

    if crop_df.shape[0] < 20:
        print(f"⚠️ Skipping {crop}, not enough data ({crop_df.shape[0]} rows).")
        continue

    X = crop_df[[cols["area"], cols["production"]]]
    y = crop_df[cols["yield"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=42))
    ])

    search = RandomizedSearchCV(
        rf_pipeline, rf_params, n_iter=10,
        scoring="r2", cv=3, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    print(f"✅ {crop} - R²: {r2_score(y_test, y_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    # Save yield model
    joblib.dump(search.best_estimator_, f"{crop.lower()}_yield_model.pkl")
    print(f"💾 Saved: {crop.lower()}_yield_model.pkl")

    # Save production model (train separately: area + yield → production)
    X_prod = crop_df[[cols["area"], cols["yield"]]]
    y_prod = crop_df[cols["production"]]

    search_prod = RandomizedSearchCV(
        rf_pipeline, rf_params, n_iter=10,
        scoring="r2", cv=3, random_state=42, n_jobs=-1
    )
    search_prod.fit(X_prod, y_prod)

    joblib.dump(search_prod.best_estimator_, f"{crop.lower()}_prod_model.pkl")
    print(f"💾 Saved: {crop.lower()}_prod_model.pkl")

    # Save yield stats
    crop_stats[crop] = {
        "min_yield": float(y.min()),
        "max_yield": float(y.max()),
        "avg_yield": float(y.mean())
    }

# Save stats JSON
with open("crop_yield_stats.json", "w") as f:
    json.dump(crop_stats, f, indent=4)

try:
    rec_df = pd.read_csv("data.csv")
    required_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    if not all(col in rec_df.columns for col in required_columns):
        raise ValueError("data.csv is missing required columns for crop classifier")

    rec_df = rec_df.dropna(subset=required_columns)
    rec_df["NPK_RATIO"] = rec_df["N"] / (rec_df["P"] + rec_df["K"] + 1e-6)
    rec_df["WEATHER_INDEX"] = rec_df["temperature"] * rec_df["humidity"] / 100 + rec_df["rainfall"] / 100

    feature_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "NPK_RATIO", "WEATHER_INDEX"]
    X_rec = rec_df[feature_columns]
    y_rec = rec_df["label"]

    label_encoder = LabelEncoder()
    y_rec_enc = label_encoder.fit_transform(y_rec)

    X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(
        X_rec, y_rec_enc, test_size=0.2, random_state=42, stratify=y_rec_enc
    )

    rec_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    rec_pipeline.fit(X_train_rec, y_train_rec)

    y_pred_rec = rec_pipeline.predict(X_test_rec)
    print(f"\n✅ Crop classifier accuracy: {accuracy_score(y_test_rec, y_pred_rec):.4f}")

    joblib.dump(rec_pipeline, "crop_classification_model.pkl")
    joblib.dump(label_encoder, "crop_label_encoder.pkl")
    print("💾 Saved crop_classification_model.pkl")
    print("💾 Saved crop_label_encoder.pkl")
except Exception as e:
    print(f"⚠️ Could not train crop classifier from data.csv: {e}")
print("\n📊 Saved crop_yield_stats.json")
