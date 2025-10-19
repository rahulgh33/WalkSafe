import joblib
from src.core.safe_path_router import SafePathRouter
from flask import Flask, request, jsonify
from flask import send_from_directory
import os

app = Flask(__name__)

# Preload model + scaler once at startup
model = joblib.load("models/safety_score_rf_model.pkl")
scaler = joblib.load("models/safety_score_scaler.pkl")

# serve built frontend from repo-root/web
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # return asset if exists, otherwise fallback to index.html for SPA routing
    if path and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return "Frontend not built. Run `npm run build` in frontend/.", 404

@app.route("/route", methods=["GET"])
def route():
    try:
        start_lat = float(request.args.get("start_lat"))
        start_lon = float(request.args.get("start_lon"))
        end_lat = float(request.args.get("end_lat"))
        end_lon = float(request.args.get("end_lon"))
        lam = float(request.args.get("lambda", 1000))

        router = SafePathRouter(
            start_coords=(start_lat, start_lon),
            end_coords=(end_lat, end_lon),
            model=model,
            scaler=scaler,
            dist_meters=800,
            assign_scores=False  # use precomputed
        )
        path = router.get_path(lambda_val=lam)
        stats = router.get_path_stats(path)

        return jsonify({
            "lambda": lam,
            "distance_m": stats["distance_m"],
            "avg_safety_score": stats["avg_safety_score"],
            "num_nodes": len(path),
            "path": [
                {
                    "lat": router.G.nodes[n]['y'],
                    "lon": router.G.nodes[n]['x'],
                    "score": router.G.nodes[n].get("safety_score", 0.5)
                }
                for n in path
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/route_map", methods=["GET"])
def route_map():
    from folium import Map, PolyLine, Marker, Icon, Element

    try:
        start_lat = float(request.args.get("start_lat"))
        start_lon = float(request.args.get("start_lon"))
        end_lat = float(request.args.get("end_lat"))
        end_lon = float(request.args.get("end_lon"))
        lambda_values = [0, 10, 100, 1000, 10000]
        colors = ["red", "orange", "yellow", "green", "blue"]

        router = SafePathRouter(
            start_coords=(start_lat, start_lon),
            end_coords=(end_lat, end_lon),
            model=model,
            scaler=scaler,
            dist_meters=800,
            assign_scores=False
        )

        # Base map
        m = Map(location=[start_lat, start_lon], zoom_start=15)
        Marker((start_lat, start_lon), popup="Start", icon=Icon(color="green")).add_to(m)
        Marker((end_lat, end_lon), popup="End", icon=Icon(color="red")).add_to(m)

        for lam, color in zip(lambda_values, colors):
            path = router.get_path(lambda_val=lam)
            coords = [(router.G.nodes[n]['y'], router.G.nodes[n]['x']) for n in path]
            PolyLine(coords, color=color, weight=5, opacity=0.8,
                     tooltip=f"λ = {lam}").add_to(m)

        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 20px; left: 20px; width: 180px;
             height: 140px; border:2px solid grey; z-index:9999; font-size:14px;
             background-color:white; padding: 10px;">
             <b>Lambda Legend</b><br>
             <i style="color:red;">●</i> λ = 0<br>
             <i style="color:orange;">●</i> λ = 10<br>
             <i style="color:yellow;">●</i> λ = 100<br>
             <i style="color:green;">●</i> λ = 1000<br>
             <i style="color:blue;">●</i> λ = 10000<br>
        </div>
        """
        m.get_root().html.add_child(Element(legend_html))
        return m.get_root().render()

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import sys
    port = 5000
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    app.run(host="0.0.0.0", port=port)

