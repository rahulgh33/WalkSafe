import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw features and labels
features = pd.read_csv("features_raw.csv")
scores = pd.read_csv("safety_scores.csv")

# Merge on Location
df = pd.merge(features, scores, on="Location")

# Define features (you can add/remove features as needed)
X = df[[
    "CrimeCount",
    "WeightedCrimeSum",
    "NumPOIs",
    "NumLitPOIs",
    "LightingRatio",
    "DistToHelp",
    "HelpScore"
]]
y = df["Score"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Predict
preds = tree.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"✅ MSE: {mse:.4f}")

# --- Plot 1: Actual vs Predicted ---
plt.figure(figsize=(6, 4))
plt.scatter(y_test, preds, c="blue", label="Predicted vs Actual")
plt.plot([0, 1], [0, 1], "r--", label="Ideal Prediction")
plt.xlabel("Actual Safety Score")
plt.ylabel("Predicted Safety Score")
plt.title("Actual vs Predicted Safety Score")     # instead of 📈plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Residual Distribution ---
residuals = y_test - preds
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=10, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Residual Error Distribution")          # instead of 🧮plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Plot 3: Feature Importance ---
importance = tree.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
plt.barh(features, importance, color="teal")
plt.title("Feature Importance (Decision Tree)")   # instead of 🌟plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
