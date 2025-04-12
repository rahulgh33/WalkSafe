import requests
import osmnx as ox
from shapely.geometry import Point
from geopy.distance import geodesic
from datetime import datetime, timedelta
from collections import defaultdict

# --- CONFIGURATION ---
RADIUS_METERS = 200
DAYS_BACK = 60
MAX_HELP_DISTANCE = 1000
MAX_DENSITY_SCORE = 1.0

CRIME_WEIGHTS = {
    "HOMICIDE": 1.0,
    "CRIM SEXUAL ASSAULT": 0.95,
    "ROBBERY": 0.9,
    "BATTERY": 0.8,
    "ASSAULT": 0.75
}

locations = [
    ("Loop", 41.8826091, -87.6279131),
    ("Lincoln Park", 41.922703, -87.651674),
    ("UChicago", 41.789703, -87.602413),
    ("Garfield Park", 41.880230, -87.726355),
    ("95th CTA", 41.722377, -87.624342),
    ("Belmont Harbor", 41.941553, -87.638756),
    ("Austin", 41.894600, -87.755800)
]

emergency_tags = {"amenity": ["police", "hospital", "fire_station"]}
poi_tags = {"amenity": True, "shop": True, "leisure": True, "tourism": True}

def get_min_distance(lat, lon, gdf):
    user_point = (lat, lon)
    min_dist = float("inf")
    for _, row in gdf.iterrows():
        try:
            if row.geometry.geom_type == 'Point':
                target = (row.geometry.y, row.geometry.x)
            else:
                target = (row.geometry.centroid.y, row.geometry.centroid.x)
            dist = geodesic(user_point, target).meters
            if dist < min_dist:
                min_dist = dist
        except:
            continue
    return round(min_dist, 1) if min_dist < float("inf") else None

def compute_crime_score_density(weighted_sum, poi_count):
    crime_density = weighted_sum / (poi_count + 5)
    return min(round(crime_density, 2), MAX_DENSITY_SCORE)

# --- Output Header ---
print(f"{'Location':<20} {'CrimeScore':<12} {'Lighting':<10} {'HelpScore'}")
print("-" * 50)

# --- Main loop ---
for name, lat, lon in locations:
    # Step 1: Crime query
    date_since = (datetime.now() - timedelta(days=DAYS_BACK)).isoformat()
    crime_api = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    params = {
        "$where": f"""
            within_circle(location, {lat}, {lon}, {RADIUS_METERS})
            AND date > '{date_since}'
        """,
        "$limit": 1000
    }
    crime_data = requests.get(crime_api, params=params).json()

    weighted_sum = 0
    incident_count = 0
    for crime in crime_data:
        ctype = crime.get("primary_type", "").upper()
        description = crime.get("description", "").upper()
        if ctype in CRIME_WEIGHTS:
            if (ctype == "BATTERY" or ctype == "ASSAULT") and "AGGRAVATED" not in description:
                continue
            weight = CRIME_WEIGHTS[ctype]
            weighted_sum += weight
            incident_count += 1

    # Step 2: POIs
    poi_gdf = ox.features_from_point((lat, lon), tags=poi_tags, dist=RADIUS_METERS)
    poi_count = len(poi_gdf)

    # Step 3: Lighting score from likely-lit POIs
    lit_poi_kinds = ["restaurant", "fast_food", "bar", "cafe", "pub", "pharmacy", "hospital", "police", "bus_station"]
    lit_poi_count = 0
    for _, row in poi_gdf.iterrows():
        for key in ["amenity", "shop", "leisure", "tourism"]:
            if key in row and str(row[key]).lower() in lit_poi_kinds:
                lit_poi_count += 1
                break

    lighting_score = min(round(lit_poi_count / 10, 2), 1.0)
    crime_score = compute_crime_score_density(weighted_sum, poi_count)

    # Step 4: Emergency help score
    emergency_gdf = ox.features_from_point((lat, lon), tags=emergency_tags, dist=2000)
    dist_to_help = get_min_distance(lat, lon, emergency_gdf)
    help_score = min(dist_to_help / MAX_HELP_DISTANCE, 1.0) if dist_to_help else 1.0

    # Final clean output
    print(f"{name:<20} {crime_score:<12.2f} {lighting_score:<10.2f} {help_score:.2f}")
