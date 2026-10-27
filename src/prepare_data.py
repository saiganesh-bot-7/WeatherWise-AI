from pathlib import Path
import pandas as pd

# Find the project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Input and output file paths
INPUT_FILE = BASE_DIR / "data" / "raw" / "india_weather.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "india_weather_clean.csv"

# Check that the raw dataset exists
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Run python src/fetch_weather.py first."
    )

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove rows with missing city or date
df = df.dropna(subset=["date", "city"])

# Sort by city and date
df = df.sort_values(["city", "date"])

# Remove duplicate city-date records
df = df.drop_duplicates(subset=["city", "date"])

# Weather columns to clean
weather_columns = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
    "cloud_cover_mean",
]

# Convert weather measurements to numeric values
for column in weather_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nMissing values before cleaning:")
print(df[weather_columns].isna().sum())

# Remove rows with missing weather measurements
df = df.dropna(subset=weather_columns).copy()

# Add date features
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

# Create processed-data directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nData preparation completed successfully!")
print("Total rows:", len(df))
print("Total cities:", df["city"].nunique())
print("Saved dataset:", OUTPUT_FILE)

print("\nRecords per city:")
print(df.groupby("city").size())