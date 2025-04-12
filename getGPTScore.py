import google.generativeai as genai
import time

# Set your API key here
genai.configure(api_key="AIzaSyB3fXklX-tB_ggUDKqVG8R0VNPvJckAilE")

# Use the Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Prompt function
def generate_prompt(lat, lon):
    return (
        f"You are a public safety expert. Please evaluate how safe is the area around latitude {lat:.6f}, longitude {lon:.6f} to walk alone at night? It is a place in chicago. Find where the coordinates are in chicago"
        f"Consider crime rates, lighting, and emergency services. Return a safety score from 0 (very unsafe) to 1 (very safe). "
        f"Only return the number."
    )

# Get safety score
def get_safety_score(lat, lon):
    try:
        prompt = generate_prompt(lat, lon)
        response = model.generate_content(prompt)
        content = response.text.strip()
        return float(content)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

# Test locations
locations = [
    ("Loop", 41.8826091, -87.6279131),
    ("Garfield Park", 41.880230, -87.726355),
    ("Lincoln Park", 41.922703, -87.651674),
    ("UChicago", 41.789703, -87.602413),
    ("Austin", 41.894600, -87.755800),
]

# Output
print(f"{'Location':<15} {'Latitude':<10} {'Longitude':<11} {'Gemini Safety Score'}")
print("-" * 60)

for name, lat, lon in locations:
    score = get_safety_score(lat, lon)
    print(f"{name:<15} {lat:<10.6f} {lon:<11.6f} {score}")
    time.sleep(1.1)
