import joblib
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from safe_path_router import SafePathRouter  # Update this if needed

# --- Load Model and Scaler ---
model = joblib.load("safety_score_rf_model.pkl")
scaler = joblib.load("safety_score_scaler.pkl")

# --- Define Start & End Coordinates ---
start_coords = (41.7796, -87.6636)  # West Garfield Park
end_coords = (41.7681, -87.6435)    # Austin

# --- Estimate Coverage Distance ---
D = geodesic(start_coords, end_coords).meters
dist_meters = 400

# --- Initialize Router ---
router = SafePathRouter(
    start_coords=start_coords,
    end_coords=end_coords,
    model=model,
    scaler=scaler,
    dist_meters=dist_meters,
    assign_scores=True
)

# --- Lambda values & colors ---
lambda_values = [0, 10, 100, 1000, 10000]
colors = ['red', 'orange', 'yellow', 'green', 'blue']
path_data = []

# === 🔽 Start of Replaced Section 🔽 ===

# --- Get initial map location ---
initial_path = router.get_path(lambda_val=lambda_values[0])
start_latlon = (router.G.nodes[initial_path[0]]['y'], router.G.nodes[initial_path[0]]['x'])
end_latlon = (router.G.nodes[initial_path[-1]]['y'], router.G.nodes[initial_path[-1]]['x'])
m = folium.Map(location=start_latlon, zoom_start=15)

# --- Add Start and End Markers ---
folium.Marker(start_latlon, popup='Start', icon=folium.Icon(color='green')).add_to(m)
folium.Marker(end_latlon, popup='End', icon=folium.Icon(color='red')).add_to(m)

# --- Draw all lambda paths ---
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

# --- Add Legend ---
legend_html = """
<div style="position: fixed;
     bottom: 20px; left: 20px; width: 180px; height: 140px;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;">
     <b>Path Lambda Legend</b><br>
     <i style="color:red;">●</i> Lambda 0<br>
     <i style="color:orange;">●</i> Lambda 10<br>
     <i style="color:yellow;">●</i> Lambda 100<br>
     <i style="color:green;">●</i> Lambda 1000<br>
     <i style="color:blue;">●</i> Lambda 10000<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# --- Save Map ---
m.save("multi_lambda_paths_map.html")
print("🗺️  Map with legend and markers saved to 'multi_lambda_paths_map.html'")

# === 🔼 End of Replaced Section 🔼 ===

# --- Display Stats Table ---
df = pd.DataFrame(path_data)
print("\n📊 Lambda Comparison Table:\n")
print(df.to_string(index=False))