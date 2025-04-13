import requests
import osmnx as ox
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta

# --- CONSTANTS ---
RADIUS_METERS = 200
DAYS_BACK = 60
MAX_HELP_DISTANCE = 1000
AREA_KM2 = np.pi * (RADIUS_METERS / 1000) ** 2

CRIME_TYPES = [
    "HOMICIDE", "CRIM SEXUAL ASSAULT", "ROBBERY", "BATTERY", "ASSAULT",
    "THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "ARSON", "KIDNAPPING"
]

EMERGENCY_TAGS = {"amenity": ["police", "hospital", "fire_station"]}
POI_TAGS = {"amenity": True, "shop": True, "leisure": True, "tourism": True}
LIT_KINDS = {"restaurant", "fast_food", "bar", "cafe", "pub", "pharmacy", "hospital", "police", "bus_station"}

# --- UTILITY ---
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
    return round(min_dist, 1) if min_dist < float("inf") else MAX_HELP_DISTANCE

# --- MAIN FUNCTION ---
def compute_features(lat, lon):
    # --- 1. Crime Data ---
    date_since = (datetime.now() - timedelta(days=DAYS_BACK)).isoformat()
    crime_api = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    params = {
        "$where": f"""
            within_circle(location, {lat}, {lon}, {RADIUS_METERS})
            AND date > '{date_since}'
        """,
        "$limit": 1000
    }
    try:
        crime_data = requests.get(crime_api, params=params).json()
    except:
        crime_data = []

    crime_counts = {ctype: 0 for ctype in CRIME_TYPES}
    gun_crimes = 0

    for crime in crime_data:
        if isinstance(crime, dict):
            ctype = crime.get("primary_type", "").upper()
            desc = crime.get("description", "").upper()
            if ctype in crime_counts:
                crime_counts[ctype] += 1
            if any(w in desc for w in ["HANDGUN", "FIREARM", "GUN", "SHOT"]):
                gun_crimes += 1

    # --- 2. POIs ---
    try:
        poi_gdf = ox.features_from_point((lat, lon), tags=POI_TAGS, dist=RADIUS_METERS)
    except ox._errors.InsufficientResponseError:
        poi_gdf = pd.DataFrame(columns=["geometry"])
    poi_count = len(poi_gdf)

    lit_poi_count = 0
    for _, row in poi_gdf.iterrows():
        for key in ["amenity", "shop", "leisure", "tourism"]:
            if key in row and str(row[key]).lower() in LIT_KINDS:
                lit_poi_count += 1
                break

    poi_density = round(poi_count / AREA_KM2, 2)
    lit_density = round(lit_poi_count / AREA_KM2, 2)

    # --- 3. Emergency Help Distance ---
    try:
        emergency_gdf = ox.features_from_point((lat, lon), tags=EMERGENCY_TAGS, dist=2000)
        dist_to_help = get_min_distance(lat, lon, emergency_gdf)
    except:
        dist_to_help = MAX_HELP_DISTANCE

    # --- 4. Return Feature Dict ---
    features = {
        "Latitude": lat,
        "Longitude": lon,
        "NumPOIs": poi_count,
        "NumLitPOIs": lit_poi_count,
        "POIDensity": poi_density,
        "LitPOIDensity": lit_density,
        "DistToHelp": dist_to_help,
        "GunCrimes": gun_crimes,
    }
    features.update(crime_counts)

    return features
