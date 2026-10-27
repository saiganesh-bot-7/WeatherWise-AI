from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# Find the main project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset output location
OUTPUT_FILE = (
    BASE_DIR / "data" / "raw" / "india_weather.csv"
)

# Six Indian cities
CITIES = {
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Jaipur": (26.9124, 75.7873),
}

# Historical period
START_DATE = "2023-01-01"

# Avoid requesting data for the most recent few days,
# which may not yet be available.
END_DATE = (
    date.today() - timedelta(days=5)
).isoformat()

# Weather variables
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
    "cloud_cover_mean",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city(city, latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "kmh",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    weather_data = response.json()

    if "daily" not in weather_data:
        raise ValueError(
            f"No weather data received for {city}"
        )

    df = pd.DataFrame(weather_data["daily"])

    df = df.rename(
        columns={"time": "date"}
    )

    df["city"] = city
    df["latitude"] = latitude
    df["longitude"] = longitude

    return df


def main():

    all_cities = []

    for city, coordinates in CITIES.items():

        latitude, longitude = coordinates

        print(f"Downloading weather for {city}...")

        try:
            city_data = fetch_city(
                city,
                latitude,
                longitude,
            )

            all_cities.append(city_data)

            print(
                f"Successfully downloaded {city}: "
                f"{len(city_data)} records"
            )

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            print(
                f"Could not download {city}: {error}"
            )

    if not all_cities:
        raise RuntimeError(
            "No weather data was downloaded. "
            "Check your internet connection and try again."
        )

    # Combine data from all cities
    df = pd.concat(
        all_cities,
        ignore_index=True,
    )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["city", "date"]
    )

    df = df.drop_duplicates(
        subset=["city", "date"]
    )

    # Create the output directory if necessary
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nDATA COLLECTION COMPLETED!")
    print(f"Dataset saved at: {OUTPUT_FILE}")
    print(f"Total records: {len(df)}")
    print(f"Total cities: {df['city'].nunique()}")

    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    main()