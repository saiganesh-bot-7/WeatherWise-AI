from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR / "data" / "processed" / "india_weather_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"
RESULTS_FILE = (
    BASE_DIR / "data" / "processed" / "test_predictions.csv"
)

# Load cleaned dataset
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["city", "date"]).copy()

# Create tomorrow's maximum temperature target
df["target_max_temp"] = (
    df.groupby("city")["temperature_2m_max"].shift(-1)
)

# Get the date associated with the target
df["target_date"] = df.groupby("city")["date"].shift(-1)

# Keep consecutive-day observations only
df = df[
    df["target_date"] == df["date"] + pd.Timedelta(days=1)
].copy()

# Input features
features = [
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

# Remove incomplete rows
df = df.dropna(subset=features + ["target_max_temp"]).copy()

if df.empty:
    raise ValueError(
        "No training rows available. Check the cleaned dataset."
    )

# Split by date to avoid randomly mixing past and future
unique_dates = np.sort(df["date"].unique())

if len(unique_dates) < 30:
    raise ValueError(
        "Not enough dates for a reliable chronological split."
    )

split_index = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_index]

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

X_train = train[features]
y_train = train["target_max_temp"]

X_test = test[features]
y_test = test["target_max_temp"]

# Build Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

# Train
print("Training Random Forest model...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Baseline: tomorrow's temperature equals today's temperature
baseline_predictions = test["temperature_2m_max"].to_numpy()

# Evaluation
model_mae = mean_absolute_error(y_test, predictions)
baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

model_rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\n========== MODEL EVALUATION ==========")
print("Training observations:", len(train))
print("Testing observations:", len(test))
print("Test period starts:", test["date"].min().date())

print(f"\nRandom Forest MAE: {model_mae:.2f} °C")
print(f"Persistence MAE:   {baseline_mae:.2f} °C")

print(f"\nRandom Forest RMSE: {model_rmse:.2f} °C")
print(f"Persistence RMSE:   {baseline_rmse:.2f} °C")

if model_mae < baseline_mae:
    print("\nRandom Forest improved on the MAE baseline.")
else:
    print("\nRandom Forest did not improve on the MAE baseline.")

# Save trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features,
    },
    MODEL_DIR / "max_temperature_model.joblib",
)

# Save test predictions
results = test[
    ["date", "city", "target_max_temp"]
].copy()

results["predicted_max_temp"] = predictions
results["baseline_max_temp"] = baseline_predictions

results.to_csv(RESULTS_FILE, index=False)

print("\nModel saved successfully!")
print("Model path:", MODEL_DIR / "max_temperature_model.joblib")
print("Results path:", RESULTS_FILE)