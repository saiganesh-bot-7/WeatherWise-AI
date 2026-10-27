# 🌦️ WeatherWise AI — Machine Learning Weather Forecasting

WeatherWise AI is an end-to-end machine learning project that uses historical weather data to predict next-day maximum temperature and rainfall likelihood for Indian cities. It includes data preparation, model training, evaluation, and an interactive Streamlit dashboard.

## 🚀 Features

* 🌡️ Predict next-day maximum temperature using regression.
* 🌧️ Estimate rainfall probability using binary classification.
* 🏙️ Explore weather data by city.
* 📊 Visualize historical temperature and rainfall trends.
* 🤖 Compare machine learning approaches using evaluation metrics.

## 🧠 Machine Learning Models

### 1. Temperature Prediction — Regression

* **Algorithm:** Random Forest Regressor
* **Target:** Next-day maximum temperature in °C
* **Evaluation:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)

### 2. Rainfall Prediction — Classification

* **Algorithm:** Random Forest Classifier
* **Target:** Whether next-day precipitation reaches at least 1 mm
* **Evaluation:** Accuracy, precision, recall, F1-score, and Brier score

## 🛠️ Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* Plotly
* Streamlit
* Matplotlib and Seaborn

## 📁 Project Structure

```text
WeatherWise-AI/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── max_temperature_model.joblib
│   └── rainfall_model.joblib
├── src/
│   ├── fetch_weather.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── rainfall_model.py
│   └── visualize_results.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd WeatherWise-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it on Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## ▶️ Run the Dashboard

From the project root, run:

```powershell
python -m streamlit run app/dashboard.py
```

Streamlit will provide a local address where you can view the dashboard.

## 📈 Evaluation

The project evaluates regression and classification separately. Actual scores should be taken from the model training output and reported here once verified.

## ⚠️ Limitations

* Predictions depend on the quality and coverage of the historical dataset.
* Historical patterns may not capture sudden weather changes.
* Rainfall probability is a model estimate, not a guarantee.
* This project is educational and should not replace official weather warnings.

## 👨‍💻 Author

Developed as an educational machine learning project demonstrating regression, classification, data analysis, model evaluation, and interactive dashboard development.
