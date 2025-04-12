import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 1. Load and Merge Data
# --------------------------
# Update these file names to match your actual feature and ground truth file paths.
features_file = "features_final.csv"    # File with scraped feature data
ground_truths_file = "safety_scores.csv"  # File with updated ground truths

# Load the datasets
features = pd.read_csv(features_file)
ground_truths = pd.read_csv(ground_truths_file)

# Ensure that the 'Location' column matches exactly across both files.
df = pd.merge(features, ground_truths, on="Location")

# Optionally, rename the ground truth column for consistency.
df.rename(columns={"SafetyScore": "Score"}, inplace=True)

print("Merged Data Preview:")
print(df.head())
print("\nData Summary:")
print(df.describe())

# --------------------------
# 2. Compute Correlation Matrix
# --------------------------
# Use only numeric columns for computing correlations.
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix for Numeric Features and Safety Score")
plt.tight_layout()
plt.show()

# --------------------------
# 3. Scatter Plots for EDA
# --------------------------
# Identify feature columns for analysis. We exclude non-numeric ones like 'Location'.
feature_cols = [col for col in df.columns if col not in ["Location"]]

# Generate scatter plots for each numeric feature versus Safety Score (excluding the target itself)
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]) and col != "Score":
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df["Score"])
        plt.title(f"Safety Score vs {col}")
        plt.xlabel(col)
        plt.ylabel("Safety Score")
        plt.tight_layout()
        plt.show()

# --------------------------
# 4. Model Training and Evaluation
# --------------------------
# Select numeric features for modeling (excluding the target Score)
model_features = df.select_dtypes(include=[np.number]).drop(columns=["Score"])
X = model_features
y = df["Score"]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Standardize features (useful if you later try models sensitive to scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Baseline Random Forest Model
baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_preds = baseline_model.predict(X_test)

print("\nBaseline Model Performance:")
print(f"MSE: {mean_squared_error(y_test, baseline_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_test, baseline_preds):.4f}")
print(f"R²: {r2_score(y_test, baseline_preds):.4f}")

# --------------------------
# 5. Hyperparameter Tuning with GridSearchCV
# --------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

# Hyperparameter Tuning and Improved Model Evaluation
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid,
                           cv=5,
                           scoring="r2",
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)
print("Best CV R² Score:", grid_search.best_score_)

improved_model = grid_search.best_estimator_
improved_preds = improved_model.predict(X_test)

print("\nImproved Model Performance:")
print(f"MSE: {mean_squared_error(y_test, improved_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_test, improved_preds):.4f}")
print(f"R²: {r2_score(y_test, improved_preds):.4f}")

# ==================================================================
# Further Diagnostic Analysis: Repeated K-Fold and Learning Curves
# ==================================================================

from sklearn.model_selection import RepeatedKFold, learning_curve

# 1. Robust Cross-Validation: Using RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_scores = cross_val_score(improved_model, X_scaled, y, cv=rkf, scoring="r2")
print("\nRepeated K-Fold CV R² Scores:", cv_scores)
print("Mean Repeated CV R² Score:", np.mean(cv_scores))

# 2. Learning Curve: Visualizing Model Performance vs. Training Size
train_sizes, train_scores, test_scores = learning_curve(
    improved_model, X_scaled, y, cv=5, scoring="r2", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="CV score")
plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# ==================================================================
# Feature Importance Analysis
# ==================================================================

importance = improved_model.feature_importances_
feature_names = X.columns
feat_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_importance_df, x="Importance", y="Feature")
plt.title("Feature Importance from Improved RandomForest Model")
plt.tight_layout()
plt.show()

