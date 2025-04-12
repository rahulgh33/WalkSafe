# enhanced_getFeatures.py
import requests
import osmnx as ox
import pandas as pd
import numpy as np
from shapely.geometry import Point
from geopy.distance import geodesic
from datetime import datetime, timedelta

# --- CONFIGURATION ---
RADIUS_METERS = 200
DAYS_BACK = 60
MAX_HELP_DISTANCE = 1000
AREA_KM2 = np.pi * (RADIUS_METERS / 1000) ** 2

CRIME_TYPES = [
    "HOMICIDE", "CRIM SEXUAL ASSAULT", "ROBBERY", "BATTERY", "ASSAULT",
    "THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "ARSON", "KIDNAPPING"
]

locations = [
    ("Lincoln Park", 41.9250, -87.6500),
    ("Lakeview", 41.9400, -87.6500),
    ("Gold Coast", 41.9000, -87.6330),
    ("Old Town", 41.8930, -87.6330),
    ("West Loop", 41.8820, -87.6440),
    ("River North", 41.8920, -87.6390),
    ("Streeterville", 41.8920, -87.6190),
    ("Edgewater", 41.9830, -87.6740),
    ("Beverly", 41.7500, -87.6800),
    ("Hyde Park", 41.7940, -87.5910),
    ("Jefferson Park", 41.9720, -87.7520),
    ("Norwood Park", 41.9630, -87.8130),
    ("Edison Park", 41.9850, -87.8240),
    ("Forest Glen", 41.9900, -87.7850),
    ("Rogers Park", 42.0100, -87.6750),
    ("Portage Park", 41.9520, -87.7670),
    ("Logan Square", 41.9250, -87.6990),
    ("Avondale", 41.9280, -87.7100),
    ("Bridgeport", 41.8470, -87.6550),
    ("Mount Greenwood", 41.7700, -87.7300),
    ("Albany Park", 41.9670, -87.7090),
    ("West Town", 41.8840, -87.6970),
    ("Dunning", 41.9680, -87.7450),
    ("Pilsen", 41.8530, -87.6680),
    ("Belmont Cragin", 41.9640, -87.7630),
    ("Uptown", 41.9660, -87.6670),
    ("Back of the Yards", 41.8130, -87.6500),
    ("West Ridge", 42.0050, -87.6970),
    ("Pullman", 41.7220, -87.5790),
    ("Hegewisch", 41.6310, -87.6010),
    ("South Loop", 41.8600, -87.6250),
    ("Kenwood", 41.8050, -87.5890),
    ("Armour Square", 41.8760, -87.6380),
    ("Archer Heights", 41.8400, -87.7550),
    ("Brighton Park", 41.8150, -87.7500),
    ("Irving Park", 41.9530, -87.7610),
    ("O'Hare", 41.9740, -87.9040),
    ("Ashburn", 41.7960, -87.7440),
    ("McKinley Park", 41.8500, -87.7010),
    ("Chatham", 41.7600, -87.6230),
    ("West Garfield Park", 41.8820, -87.7200),
    ("East Garfield Park", 41.8800, -87.7100),
    ("Austin", 41.8820, -87.7310),
    ("Englewood", 41.7750, -87.6400),
    ("West Englewood", 41.7730, -87.6500),
    ("South Shore", 41.7800, -87.5800),
    ("Roseland", 41.8000, -87.5500),
    ("North Lawndale", 41.8700, -87.7100),
    ("Fuller Park", 41.8000, -87.6300),
    ("Washington Park", 41.8000, -87.6400),
    ("Grand Boulevard", 41.8100, -87.6400),
    ("Gage Park", 41.8300, -87.6600),
    ("Garfield Ridge", 41.8400, -87.7600),
    ("Clearing", 41.9400, -87.7800),
    ("Douglas", 41.8700, -87.6300),
    ("Bronzeville", 41.8000, -87.6300),
    ("Little Village", 41.8450, -87.7070),
    ("Lawndale", 41.8800, -87.7400),
    ("New City", 41.8000, -87.6400),
    ("Humboldt Park", 41.9050, -87.7010),
    ("Woodlawn", 41.7800, -87.6000),
    ("Burnside", 41.7600, -87.6400),
    ("Avalon Park", 41.7900, -87.6000),
    ("Calumet Heights", 41.7500, -87.5500),
    ("South Deering", 41.7500, -87.5800),
    ("East Side", 41.8000, -87.5700),
    ("Morgan Park", 41.7200, -87.6300),
    ("South Chicago", 41.7800, -87.5800),
    ("West Pullman", 41.7400, -87.6400),
    ("Riverdale", 41.7400, -87.6000),
    ("West Elsdon", 41.8000, -87.7300),
    ("West Lawn", 41.7800, -87.6800),
    ("Chicago Lawn", 41.8100, -87.6800),
    ("North Park", 41.9900, -87.7600),
    ("Hermosa", 41.8700, -87.7500),
    ("Sauganash", 41.9800, -87.7500),
    ("Peterson Park", 41.9320, -87.6700),
    ("Mayfair", 41.9200, -87.6800),
    ("Clearing East", 41.9500, -87.7700),
    ("Scottsdale", 41.9500, -87.7700),
    ("Beverly Hills", 41.9000, -87.6800),
    ("Galewood", 41.9200, -87.7000),
    ("Portage Park East", 41.9600, -87.7800),
    ("Avondale West", 41.9300, -87.7100),
    ("North Center", 41.9400, -87.6800),
    ("Ravenswood", 41.9800, -87.6800),
    ("Lincoln Square", 42.0100, -87.7000),
    ("Wrigleyville", 41.9400, -87.6560),
    ("Boystown", 41.9000, -87.6600),
    ("Andersonville", 42.0100, -87.6700),
    ("Chinatown", 41.8500, -87.6480),
    ("Near North Side", 41.9000, -87.6300),
    ("Near West Side", 41.8700, -87.6800),
    ("Near South Side", 41.8600, -87.6250),
    ("East Pilsen", 41.8600, -87.6700),
    ("The Island", 41.8780, -87.6300),
    ("Chicago Midway", 41.7850, -87.7520),
    ("Little Italy", 41.8790, -87.6460),
    ("Prairie District", 41.8800, -87.6300),
    ("Jackson Park Highlands", 41.7850, -87.5800),
    ("Pullman National Monument", 41.6819, -87.6083),
    ("Devon Avenue", 41.9987, -87.6978),
    ("Museum Campus", 41.8663, -87.6167),
    ("Millennium Park", 41.8826, -87.6226),
    ("Navy Pier", 41.8916, -87.6079),
    ("UIC", 41.8708, -87.6498),
    ("University of Chicago", 41.7897, -87.5996),
    ("Wrigley Field", 41.9484, -87.6553),
    ("Greektown", 41.8807, -87.6484),
    ("Southport Corridor", 41.9440, -87.6632)
]



emergency_tags = {"amenity": ["police", "hospital", "fire_station"]}
poi_tags = {"amenity": True, "shop": True, "leisure": True, "tourism": True}

data = []

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

# --- Main loop ---
print(f"{'Location':<20} {'TotalCrimes':<12} {'POIs':<6} {'LitPOIs':<8} {'HelpDist':<10}")
print("-" * 60)

for name, lat, lon in locations:
    print(f"🔍 Processing {name}")

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

    # Step 2: POIs
    try:
        poi_gdf = ox.features_from_point((lat, lon), tags=poi_tags, dist=RADIUS_METERS)
    except ox._errors.InsufficientResponseError:
        poi_gdf = pd.DataFrame(columns=["geometry"])
    poi_count = len(poi_gdf)

    lit_kinds = {"restaurant", "fast_food", "bar", "cafe", "pub", "pharmacy", "hospital", "police", "bus_station"}
    lit_poi_count = 0
    for _, row in poi_gdf.iterrows():
        for key in ["amenity", "shop", "leisure", "tourism"]:
            if key in row and str(row[key]).lower() in lit_kinds:
                lit_poi_count += 1
                break

    poi_density = round(poi_count / AREA_KM2, 2)
    lit_density = round(lit_poi_count / AREA_KM2, 2)

    # Step 3: Help distance
    try:
        emergency_gdf = ox.features_from_point((lat, lon), tags=emergency_tags, dist=2000)
        dist_to_help = get_min_distance(lat, lon, emergency_gdf)
    except:
        dist_to_help = MAX_HELP_DISTANCE

    # Step 4: Save features
    record = {
        "Location": name,
        "NumPOIs": poi_count,
        "NumLitPOIs": lit_poi_count,
        "POIDensity": poi_density,
        "LitPOIDensity": lit_density,
        "DistToHelp": dist_to_help,
        "GunCrimes": gun_crimes,
    }
    record.update(crime_counts)
    data.append(record)
    print(f"{name:<20} {sum(crime_counts.values()):<12} {poi_count:<6} {lit_poi_count:<8} {dist_to_help:>6.1f}")

pd.DataFrame(data).to_csv("features_final.csv", index=False)
print("✅ Final features saved to features_final.csv")
