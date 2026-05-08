# 🌦️ AI Weather Trend Forecasting Project

## 🚀 PM Accelerator Mission
PM Accelerator aims to help aspiring professionals gain hands-on experience by building real-world AI products in a collaborative environment. This project aligns with that mission by applying data science and machine learning techniques to solve real-world weather forecasting problems.

---

## 📌 Project Overview
This project focuses on analyzing global weather data and forecasting temperature trends using machine learning models. It includes end-to-end data science workflow: data preprocessing, exploratory data analysis (EDA), anomaly detection, spatial visualization, and predictive modeling.

---

## 📊 Dataset
- Source: Kaggle - Global Weather Repository  
- Features: 40+ weather-related variables (temperature, humidity, pressure, air quality, wind speed, precipitation, etc.)  
- Data Type: Time-series weather observations across global locations  

---

## ⚙️ Methodology

### 1. Data Cleaning & Preprocessing
- Handled missing values using median imputation and forward filling  
- Converted and sorted timestamps (last_updated) for time-series analysis  
- Applied time-based train/test split to avoid data leakage  

---

### 2. Exploratory Data Analysis (EDA)
- Temperature trend over time  
- Precipitation trend analysis  
- Correlation matrix between weather and air quality features  

---

### 3. Anomaly Detection
- Applied Isolation Forest to detect abnormal weather patterns  
- Visualized anomalies in temperature vs precipitation  

---

### 4. Spatial Analysis
- Visualized global temperature distribution using geographic mapping  
- Identified regional differences in weather patterns  

---

### 5. Machine Learning Models

Models used:
- Linear Regression  
- Random Forest  
- XGBoost  
- Weighted Ensemble  

Evaluation metrics:
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

---

## 📈 Results

- Random Forest and XGBoost achieved the best performance (R² ≈ 0.52)  
- Linear Regression underperformed due to non-linear relationships  
- Ensemble model did not significantly outperform individual models  

---

## 💡 Key Insights

- Weather forecasting is challenging without temporal (lag) features  
- Pressure and humidity are strong predictors of temperature  
- Extreme precipitation events were detected as anomalies  
- Weather patterns vary significantly across geographic regions  

---

## 🔍 Future Improvements

- Add lag features for better time-series forecasting  
- Explore time-series models (ARIMA, LSTM)  
- Improve ensemble diversity for better performance  

---

## 🗂️ Project Structure


```plaintext

weather_trend_forecasting/

│

├── data/                  # Raw dataset

├── output/                # Generated plots and analysis results

├── advanced_weather_analysis.py

└── README.md

```

## ▶️ How to Run

bash pip install -r requirements.txt python advanced_weather_analysis.py 

---

## 📁 Output Files

The project generates:
- Temperature & precipitation trend plots  
- Correlation matrix  
- Anomaly detection visualization  
- Spatial analysis map (HTML)  
- Feature importance plots  
- Model performance results (CSV/JSON)  

---

## 🎥 Demo Video
https://youtu.be/aIwE-XCv0o4

