from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "india_weather_clean.csv"
TEMP_MODEL_PATH = ROOT / "models" / "max_temperature_model.joblib"
RAIN_MODEL_PATH = ROOT / "models" / "rainfall_model.joblib"

st.set_page_config(
    page_title="WeatherWise AI",
    page_icon="🌦️",
    layout="wide",
)

st.title("🌦️ WeatherWise AI")
st.subheader("Machine Learning Weather Forecasting")
st.write(
    "Explore historical weather data and machine learning predictions "
    "for Indian cities."
)

# Check required files
for path in [DATA_PATH, TEMP_MODEL_PATH, RAIN_MODEL_PATH]:
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_resource
def load_models(temp_path, rain_path):
    temperature_bundle = joblib.load(temp_path)
    rainfall_bundle = joblib.load(rain_path)

    # Temperature model was saved as a dictionary.
    if isinstance(temperature_bundle, dict):
        temperature_model = temperature_bundle["model"]
        temperature_features = temperature_bundle["features"]
    else:
        temperature_model = temperature_bundle
        temperature_features = list(
            getattr(temperature_model, "feature_names_in_", [])
        )

    return (
        temperature_model,
        temperature_features,
        rainfall_bundle,
    )


df = load_data(str(DATA_PATH))

(
    temperature_model,
    temperature_features,
    rainfall_bundle,
) = load_models(
    str(TEMP_MODEL_PATH),
    str(RAIN_MODEL_PATH),
)

rainfall_model = rainfall_bundle["model"]
rain_input_features = rainfall_bundle["input_features"]
rain_feature_columns = rainfall_bundle["feature_columns"]

# City selection
st.sidebar.header("Forecast Settings")

cities = sorted(df["city"].dropna().unique().tolist())

if not cities:
    st.error("No cities found in the dataset.")
    st.stop()

city = st.sidebar.selectbox("Select a city", cities)

city_data = df[df["city"] == city].sort_values("date")

if city_data.empty:
    st.error("No weather records found for this city.")
    st.stop()

latest = city_data.iloc[-1]

# Prepare temperature model input
base_temp_input = {
    "temperature_2m_max": latest["temperature_2m_max"],
    "temperature_2m_min": latest["temperature_2m_min"],
    "temperature_2m_mean": latest["temperature_2m_mean"],
    "precipitation_sum": latest["precipitation_sum"],
    "relative_humidity_2m_mean": latest["relative_humidity_2m_mean"],
    "wind_speed_10m_max": latest["wind_speed_10m_max"],
    "pressure_msl_mean": latest["pressure_msl_mean"],
    "cloud_cover_mean": latest["cloud_cover_mean"],
    "month": int(latest["month"]),
    "day_of_year": int(latest["day_of_year"]),
}

temp_input = pd.DataFrame([base_temp_input])

# Add city columns if the model was trained using one-hot encoding.
temp_input["city"] = city
temp_input = pd.get_dummies(temp_input, columns=["city"])

if temperature_features:
    temp_input = temp_input.reindex(
        columns=temperature_features,
        fill_value=0,
    )

try:
    predicted_temperature = float(
        temperature_model.predict(temp_input)[0]
    )
except Exception as exc:
    st.error(f"Temperature prediction failed: {exc}")
    st.stop()

# Prepare rainfall model input
rain_input = pd.DataFrame([{
    feature: latest[feature]
    for feature in rain_input_features
}])

rain_input["city"] = city
rain_input = pd.get_dummies(rain_input, columns=["city"])
rain_input = rain_input.reindex(
    columns=rain_feature_columns,
    fill_value=0,
)

try:
    rain_probabilities = rainfall_model.predict_proba(rain_input)[0]
    classes = list(rainfall_model.classes_)

    # Find the probability belonging to the positive rain class (1).
    rain_class_index = classes.index(1)
    rain_probability = float(rain_probabilities[rain_class_index])
except Exception as exc:
    st.error(f"Rainfall prediction failed: {exc}")
    st.stop()

rain_status = (
    "Rain likely" if rain_probability >= 0.5
    else "Rain unlikely"
)

# Dashboard metrics
st.header(f"Weather Overview — {city}")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Observed Maximum Temperature",
    f"{latest['temperature_2m_max']:.1f} °C",
)

col2.metric(
    "Predicted Next-Day Maximum Temperature",
    f"{predicted_temperature:.1f} °C",
)

col3.metric(
    "Predicted Rain Probability",
    f"{rain_probability:.1%}",
)

st.info(
    f"Rainfall outlook: {rain_status}. "
    "This is a machine learning estimate, not a guarantee."
)

# Historical temperature chart
st.subheader("🌡️ Temperature History")

chart_data = city_data.tail(180)

temperature_chart = px.line(
    chart_data,
    x="date",
    y=["temperature_2m_max", "temperature_2m_min"],
    labels={
        "date": "Date",
        "value": "Temperature (°C)",
        "variable": "Measurement",
    },
    title=f"Maximum and Minimum Temperature — {city}",
)

st.plotly_chart(temperature_chart, use_container_width=True)

# Historical rainfall chart
st.subheader("🌧️ Rainfall History")

rain_chart = px.bar(
    chart_data,
    x="date",
    y="precipitation_sum",
    labels={
        "date": "Date",
        "precipitation_sum": "Rainfall (mm)",
    },
    title=f"Daily Rainfall — {city}",
)

st.plotly_chart(rain_chart, use_container_width=True)

# Recent records
with st.expander("View Recent Weather Records"):
    st.dataframe(
        chart_data.tail(15).sort_values(
            "date",
            ascending=False,
        ),
        use_container_width=True,
    )

st.caption(
    "WeatherWise AI is an educational machine learning project. "
    "Prediction quality depends on historical data quality and model performance."
)
