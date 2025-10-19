# WalkSafe
**Urban navigation system combining street-level safety modeling with graph-based routing.**

WalkSafe computes safer pedestrian routes by integrating machine-learned safety scores into a shortest-path search over real city street graphs.  
The system combines data on crime density, lighting, and proximity to emergency services to estimate relative safety at the street level.

---

## 🧠 Overview

WalkSafe models urban safety as a spatially varying risk field.  
Each street segment (graph node) is assigned a **safety score** predicted by a Random Forest trained on real geospatial data.  
Routing then minimizes a weighted cost function:

**Cost Function:**  
`cost(edge) = distance + λ × (1 - safety)`

By adjusting λ, users can explore the trade-off between the shortest and the safest path.

---

## 🧩 System Architecture
```
                   ┌────────────────────────────┐
                   │   Raw Geospatial Datasets   │
                   │ (Crime, Lighting, Services) │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   Feature Extraction Layer  │
                    │  (GeoPandas + OSM data)     │
                    └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Random Forest Safety Model   │
                   │ (scikit-learn, trained offline)│
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   Precomputed Node Scoring    │
                   │ (Applied to OSM street graph) │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Routing Engine (Flask API)   │
                   │  Dijkstra search w/ λ-safety  │
                   └──────────────────────────────┘
```

---

## 🧰 Tech Stack
- **Python**, **Flask**
- **scikit-learn**, **GeoPandas**, **osmnx**, **Folium**
- **Docker** for containerization  
- **GitHub Actions** for continuous integration and testing

---

## ⚙️ Development Setup

**Clone and install dependencies:**
```bash
git clone https://github.com/rahulgh33/WalkSafe.git
cd WalkSafe
pip install -r requirements.txt
```

**Run locally:**
```bash
python -m src.api.app
```

**Containerize (optional):**
```bash
docker build -t walksafe .
docker run -p 5001:5001 walksafe
```

---

## 📖 Description

WalkSafe is designed as an experimental framework for **data-driven urban navigation**, emphasizing explainability and public-safety modeling.  
Its modular design allows substitution of models, additional features (e.g., temporal crime variation), or front-end map layers for real-time interaction.
