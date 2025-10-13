import osmnx as ox
import pandas as pd
import joblib
from getFeatures import compute_features
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point

print("🧠 Loading trained model and scaler...")
model = joblib.load("safety_score_rf_model.pkl")
scaler = joblib.load("safety_score_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Define region
CENTER = (41.8781, -87.6298)
DIST_M = 300

print("📡 Downloading OSM graph...")
G = ox.graph_from_point(CENTER, dist=DIST_M, network_type="walk")
print(f"✅ Downloaded graph with {len(G.nodes)} nodes.")

results = []
print("🚀 Computing safety predictions for each node (this will take several minutes)...")

for node in tqdm(G.nodes, desc="Processing nodes"):
    y, x = G.nodes[node]['y'], G.nodes[node]['x']
    try:
        fdict = compute_features(y, x)
        X = pd.DataFrame([[fdict.get(col, 0) for col in feature_columns]], columns=feature_columns)
        X_scaled = scaler.transform(X)
        score = float(model.predict(X_scaled)[0])
    except Exception as e:
        score = 0.5  # fallback score
    results.append({"node": node, "lat": y, "lon": x, "score": score})

df = pd.DataFrame(results)
df.to_csv("precomputed_safety_scores.csv", index=False)
print(f"✅ Saved {len(df)} precomputed safety scores to precomputed_safety_scores.csv.")

