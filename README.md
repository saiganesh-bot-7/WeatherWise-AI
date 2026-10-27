# 🌦️ WeatherWise AI — Machine Learning Weather Forecasting

WeatherWise AI is a machine learning project that predicts next-day maximum temperature and the probability of rainfall using historical weather data from India.

🌐 **Live Demo:** https://weatherwise-ai-dmgunkccxzjdschpv5keg9.streamlit.app/

📂 **GitHub Repository:** https://github.com/saiganesh-bot-7/WeatherWise-AI

## Features

* 🌡️ Predicts next-day maximum temperature.
* 🌧️ Estimates the probability of rainfall.
* 📊 Displays predictions and charts in an interactive dashboard.
* 🤖 Uses Random Forest regression and classification.
* 🇮🇳 Analyzes historical Indian weather data.

## Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* Matplotlib and Seaborn
* Streamlit
* Joblib
* Git and GitHub

## Project Structure

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
│   ├── rainfall_model.py
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/saiganesh-bot-7/WeatherWise-AI.git
cd WeatherWise-AI
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app/dashboard.py
```

## Machine Learning Models

### Temperature Regression

Predicts next-day maximum temperature.

Evaluation metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Rainfall Classification

Estimates the probability of rainfall.

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Brier score

Add the actual evaluation results from your model outputs when available.

## Future Improvements

* Integrate live weather data.
* Add more Indian cities.
* Compare multiple forecasting algorithms.
* Improve model accuracy through feature engineering.
* Add more interactive charts.

## Live Demo

**Try the application:** https://weatherwise-ai-dmgunkccxzjdschpv5keg9.streamlit.app/

## Author

**Sai Ganesh**
AI/ML Student | Python | Machine Learning | Data Analysis

---

⭐ Feedback and suggestions are welcome!
