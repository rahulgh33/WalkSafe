import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------------------
features = pd.read_csv("features_final.csv")
ground_truths = pd.read_csv("safety_scores.csv")

# ---------------------------------------------------------------------
# 2. Match each feature to its nearest labeled safety location
# ---------------------------------------------------------------------
def find_nearest_score(lat, lon, gt):
    # Compute Euclidean distance in degrees (~ rough metric)
    gt["dist"] = np.sqrt((gt["Latitude"] - lat) ** 2 + (gt["Longitude"] - lon) ** 2)
    nearest_idx = gt["dist"].idxmin()
    return gt.loc[nearest_idx, "SafetyScore"]

features["SafetyScore"] = features.apply(
    lambda row: find_nearest_score(row["Latitude"], row["Longitude"], ground_truths),
    axis=1
)

print(f"✅ Assigned safety scores to {len(features)} feature rows (using nearest neighborhoods).")

# ---------------------------------------------------------------------
# 3. Prepare training data
# ---------------------------------------------------------------------
X = features.drop(columns=["SafetyScore"], errors="ignore")
y = features["SafetyScore"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ---------------------------------------------------------------------
# 4. Save artifacts
# ---------------------------------------------------------------------
joblib.dump(model, "safety_score_rf_model.pkl")
joblib.dump(scaler, "safety_score_scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("\n✅ Model, scaler, and feature list saved successfully.")

print("\n✅ Model and scaler saved successfully.")

