import osmnx as ox
import networkx as nx
import numpy as np
import folium
from getFeatures import compute_features
import pandas as pd
import time
from osmnx.simplification import consolidate_intersections
from shapely.geometry import Point


class SafePathRouter:
    def __init__(self, start_coords, end_coords, model, scaler,
                 dist_meters, network_type="walk", assign_scores=True):
        print("🔄 Initializing SafePathRouter...")
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.model = model
        self.scaler = scaler
        self.center = self._compute_midpoint(start_coords, end_coords)

        print("📡 Downloading graph...")
        self.G = ox.graph_from_point(self.center, dist=dist_meters, network_type=network_type)
        ox.distance.add_edge_lengths(self.G)

        print("🧹 Filtering to intersections...")
        largest_cc_nodes = max(nx.connected_components(self.G.to_undirected()), key=len)
        self.G = self.G.subgraph(largest_cc_nodes).copy()

        self.G = ox.project_graph(self.G)
        self.G = consolidate_intersections(self.G, tolerance=15, rebuild_graph=True)
        self.G = self.G.subgraph([n for n in self.G.nodes if len(self.G[n]) >= 3]).copy()
        print(f"✅ Final graph has {len(self.G.nodes)} intersection nodes.")

        if assign_scores:
            print("🧠 Assigning safety scores to nodes...")
            self._assign_safety_scores()
            print("✅ Node scoring complete.")

        # For Folium plotting
        self.G = ox.project_graph(self.G, to_latlong=True)

    def _compute_midpoint(self, coord1, coord2):
        lat = (coord1[0] + coord2[0]) / 2
        lon = (coord1[1] + coord2[1]) / 2
        return (lat, lon)

    def _predict_safety_score(self, lat, lon):
        features_dict = compute_features(lat, lon)

        feature_vector = pd.DataFrame([{
            "NumPOIs": features_dict["NumPOIs"],
            "NumLitPOIs": features_dict["NumLitPOIs"],
            "POIDensity": features_dict["POIDensity"],
            "LitPOIDensity": features_dict["LitPOIDensity"],
            "DistToHelp": features_dict["DistToHelp"],
            "GunCrimes": features_dict["GunCrimes"],
            "HOMICIDE": features_dict["HOMICIDE"],
            "CRIM SEXUAL ASSAULT": features_dict["CRIM SEXUAL ASSAULT"],
            "ROBBERY": features_dict["ROBBERY"],
            "BATTERY": features_dict["BATTERY"],
            "ASSAULT": features_dict["ASSAULT"],
            "THEFT": features_dict["THEFT"],
            "BURGLARY": features_dict["BURGLARY"],
            "MOTOR VEHICLE THEFT": features_dict["MOTOR VEHICLE THEFT"],
            "ARSON": features_dict["ARSON"],
            "KIDNAPPING": features_dict["KIDNAPPING"],
            "Latitude": features_dict["Latitude"],
            "Longitude": features_dict["Longitude"]
        }])

        scaled_vector = self.scaler.transform(feature_vector)
        return float(self.model.predict(scaled_vector)[0])

    def _assign_safety_scores(self):
        total = len(self.G.nodes)
        start_time = time.time()
        for i, node in enumerate(self.G.nodes):
            try:
                pt_proj = Point(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
                pt_latlon, _ = ox.projection.project_geometry(pt_proj, crs=self.G.graph["crs"], to_latlong=True)
                lat, lon = pt_latlon.y, pt_latlon.x
                score = self._predict_safety_score(lat, lon)
            except Exception as e:
                print(f"⚠️ Warning: Failed to compute safety at ({lat:.4f}, {lon:.4f}) — {e}")
                score = 0.5
            self.G.nodes[node]['safety_score'] = score

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"🧮 Processed {i + 1}/{total} nodes")

        elapsed = time.time() - start_time
        print(f"⏱️ Safety scoring took {elapsed:.1f} seconds for {total} nodes.")

    def _update_edge_weights(self, lambda_val):
        for u, v, key, data in self.G.edges(keys=True, data=True):
            dist = data.get("length", 1.0)
            safety = self.G.nodes[v].get("safety_score", 0.5)
            penalty = lambda_val * (1 - safety)
            data["custom_weight"] = dist + penalty

    def get_path(self, lambda_val=0.5):
        print(f"🚶 Computing shortest path with lambda = {lambda_val}...")
        self._update_edge_weights(lambda_val)
        orig_node = ox.distance.nearest_nodes(self.G, self.start_coords[1], self.start_coords[0])
        dest_node = ox.distance.nearest_nodes(self.G, self.end_coords[1], self.end_coords[0])
        if orig_node == dest_node:
            print("⚠️ Warning: Origin and destination snapped to same node!")
        path = nx.shortest_path(self.G, orig_node, dest_node, weight='custom_weight')
        print("✅ Shortest path found.")
        return path

    def get_path_stats(self, path):
        total_dist = sum(self.G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:]))
        avg_safety = np.mean([self.G.nodes[n].get("safety_score", 0.5) for n in path])
        return {"distance_m": total_dist, "avg_safety_score": avg_safety}

    def plot_path_folium(self, path, zoom_start=15):
        route_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in path]
        m = folium.Map(location=route_coords[0], zoom_start=zoom_start)
        folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(m)
        folium.Marker(route_coords[0], popup='Start', icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(route_coords[-1], popup='End', icon=folium.Icon(color='red')).add_to(m)
        return m