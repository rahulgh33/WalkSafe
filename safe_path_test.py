import joblib
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from safe_path_router import SafePathRouter 

# --- Load Model and Scaler ---
model = joblib.load("safety_score_rf_model.pkl")
scaler = joblib.load("safety_score_scaler.pkl")

# --- Define Start & End Coordinates (High Crime Area Test) ---
start_coords = (41.8818, -87.7295)  # Near Madison & Pulaski (Garfield Park)
end_coords = (41.8946, -87.7558)    # Austin neighborhood

# --- Graph Buffer Distance ---
D = geodesic(start_coords, end_coords).meters
dist_meters = int(D * 2.5)

# --- Initialize SafePathRouter ---
router = SafePathRouter(
    start_coords=start_coords,
    end_coords=end_coords,
    model=model,
    scaler=scaler,
    dist_meters=dist_meters,
    assign_scores=True
)

# --- Lambda Values to Try ---
lambda_values = [0, 10, 100, 1000, 10000]
colors = ['red', 'orange', 'yellow', 'green', 'blue']
path_data = []

# --- Initialize Map ---
initial_path = router.get_path(lambda_val=lambda_values[0])
start_latlon = (router.G.nodes[initial_path[0]]['y'], router.G.nodes[initial_path[0]]['x'])
m = folium.Map(location=start_latlon, zoom_start=15)

# --- Loop through Lambda Values ---
for lam, color in zip(lambda_values, colors):
    try:
        path = router.get_path(lambda_val=lam)
        stats = router.get_path_stats(path)

        coords = [(router.G.nodes[n]['y'], router.G.nodes[n]['x']) for n in path]
        folium.PolyLine(coords, color=color, weight=5, opacity=0.7,
                        tooltip=f'Lambda {lam}').add_to(m)

        unsafe_nodes = sum(router.G.nodes[n].get("safety_score", 0.5) < 0.5 for n in path)
        min_safety = min(router.G.nodes[n].get("safety_score", 0.5) for n in path)
        max_safety = max(router.G.nodes[n].get("safety_score", 0.5) for n in path)

        path_data.append({
            "Lambda": lam,
            "Distance (m)": round(stats['distance_m'], 1),
            "Avg Safety Score": round(stats['avg_safety_score'], 3),
            "Min Safety": round(min_safety, 3),
            "Max Safety": round(max_safety, 3),
            "# Unsafe Nodes": unsafe_nodes
        })

    except Exception as e:
        print(f"❌ Failed to compute path for lambda={lam}: {e}")

# --- Save Map ---
m.save("multi_lambda_paths_map.html")
print("🗺️  Saved interactive map to 'multi_lambda_paths_map.html'")

# --- Output Table ---
df = pd.DataFrame(path_data)
print("\n📊 Lambda Comparison Table:\n")
print(df.to_string(index=False))

