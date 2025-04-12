import openai
import time

# 🔐 Set your API key
openai.api_key = "your-openai-api-key"  # ← Replace this with your real key

# 🧠 Create the prompt
def generate_prompt_from_coordinates(lat, lon):
    return (
        f"How safe is the area around latitude {lat:.6f}, longitude {lon:.6f} to walk alone at night? "
        f"Give a safety score from 0 (very unsafe) to 1 (very safe). Only return the number."
    )

# 🚀 Send request to ChatGPT
def get_gpt_score_from_coordinates(lat, lon):
    prompt = generate_prompt_from_coordinates(lat, lon)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a safety evaluator AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        return float(content)
    except Exception as e:
        print(f"⚠️ Error at ({lat}, {lon}):", e)
        return None

# 🌍 Define your locations
locations = [
    ("Loop", 41.8826091, -87.6279131),
    ("Garfield Park", 41.880230, -87.726355),
    ("Lincoln Park", 41.922703, -87.651674),
    ("UChicago", 41.789703, -87.602413),
    ("Austin", 41.894600, -87.755800),
]

# 📋 Run and collect scores
print(f"{'Location':<15} {'Latitude':<10} {'Longitude':<11} {'GPT Safety Score'}")
print("-" * 55)

for name, lat, lon in locations:
    score = get_gpt_score_from_coordinates(lat, lon)
    print(f"{name:<15} {lat:<10.6f} {lon:<11.6f} {score}")
    time.sleep(1.1)  # avoid OpenAI rate limit
