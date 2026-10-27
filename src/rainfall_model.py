import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/india_weather_clean.csv"
MODEL_PATH = "models/rainfall_model.joblib"
PREDICTIONS_PATH = "data/processed/rainfall_predictions.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load cleaned weather data
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["city", "date"]).reset_index(drop=True)

# Create a next-day rainfall target for each city
df["next_date"] = df.groupby("city")["date"].shift(-1)
df["next_precipitation"] = df.groupby("city")["precipitation_sum"].shift(-1)

# Keep only consecutive days with valid rainfall data
df = df[
    (df["next_date"] == df["date"] + pd.Timedelta(days=1))
    & df["next_precipitation"].notna()
].copy()

# Rain means at least 1 mm of precipitation the next day
df["rain_tomorrow"] = (df["next_precipitation"] >= 1.0).astype(int)

# Prepare input features
feature_columns = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
    "cloud_cover_mean",
    "month",
    "day_of_year",
]

# Add city information as categorical features
X = pd.get_dummies(df[feature_columns + ["city"]], columns=["city"])
y = df["rain_tomorrow"]

# Use a chronological split rather than a random split
split_date = df["date"].quantile(0.8)
train_mask = df["date"] < split_date
test_mask = ~train_mask

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Not enough data for training and testing.")

if y_train.nunique() < 2:
    raise ValueError(
        "Training data contains only one rainfall class. "
        "Fetch more historical data or check the rainfall values."
    )

# Train the classifier
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Evaluate on unseen dates
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n=== Rainfall Prediction Model ===")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No rain", "Rain"],
        zero_division=0,
    )
)
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

if y_test.nunique() == 2:
    print(f"Brier score: {brier_score_loss(y_test, probabilities):.3f}")
else:
    print("Brier score: unavailable because the test set contains one class.")

# Save the model and predictions
joblib.dump(
    {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "input_features": feature_columns,
        "rain_threshold_mm": 1.0,
    },
    MODEL_PATH,
)

results = df.loc[X_test.index, ["date", "city"]].copy()
results["actual_rain"] = y_test
results["predicted_rain"] = predictions
results["rain_probability"] = probabilities
results["actual_precipitation_mm"] = df.loc[
    X_test.index, "next_precipitation"
]
results.to_csv(PREDICTIONS_PATH, index=False)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Predictions saved to: {PREDICTIONS_PATH}")