# IoT-Enabled Air Quality Monitoring — ML Data Analysis

**Published:** Submitted to IEEC 2026 (11th International Electrical Engineering Conference)  
**Paper:** Machine Learning Based Data Analysis of a Real-Time IoT-Enabled Environment Monitoring System  
**Authors:** Muneeba Mubarak, Ahamed Safnas, Chanduni Amaya Kumari Lokuliyana  
**Institution:** NED University of Engineering and Technology, Karachi, Pakistan

## Project Overview

Analysis of 43,148 real-time outdoor air quality readings collected from an IoT sensor node deployed at NED University (Feb–Sep 2024).

**Hardware:** Raspberry Pi 4 + PMS5003 particulate matter sensor + DHT11 temperature/humidity sensor  
**Data collection frequency:** Every 4 minutes via WiFi to MQTT broker  
**Variables monitored:** PM1, PM2.5, PM10, Temperature, Humidity

## Objectives

- Preprocess and clean large-scale IoT time-series sensor data
- Evaluate multiple ML models for PM2.5 and PM10 prediction
- Identify the best model for interpretable, production-ready air 
  quality forecasting

## Dataset

| Property | Value |
|---|---|
| Total readings | 43,148 |
| Time range | February – September 2024 |
| Variables | PM1, PM2.5, PM10, Temperature, Humidity |
| Sensor | PMS5003 Plantower + DHT11 |
| Controller | Raspberry Pi 4 |
| Collection interval | 4 minutes |

**Preprocessing steps:**
- Mixed datetime format parsing using `pd.to_datetime(format='mixed', dayfirst=True)`
- IQR-based outlier removal (temperature > 60°C flagged as artefact)
- Feature engineering: hour of day, day of week, month
- Train/test split: Months 2–6 (train) / Months 7–9 (test) — no shuffling to prevent data leakage


## Models Evaluated

### Model 1 — PCA (Principal Component Analysis)
- Used for dimensionality reduction and exploratory analysis
- Reduced 5 features to 2–3 principal components capturing ~95% variance
- Confirmed high correlation between PM1, PM2.5, PM10
- Not used as a forecasting model — deployed as visualisation tool only

### Model 2 — FB Prophet
- Time-series decomposition with temperature + humidity as  additive regressors
- Captured weekly seasonality (Saturday peak: +2.2 µg/m³) and  daily rush-hour peak (06:51–10:17)
- Accuracy: ~83% | RMSE: ~4.3
- Limitation: high uncertainty bands in July–September due to reduced data density

### Model 3 — ARIMA / SARIMA
- ARIMA(1,1,1) selected via AIC minimisation
- ADF test confirmed d=1 differencing required for stationarity
- Accuracy: ~81% | RMSE: ~4.6
- Limitation: struggled with non-stationary periods during sensor downtime

### Model 4 — Linear Regression (Final Model)
- Predicts PM2.5 and PM10 from temperature, humidity, hour, day of week
- Best balance of accuracy and interpretability for this dataset size

**Final equation:**
- PM2.5 = β₀ + β₁·Temperature + β₂·Humidity + β₃·Hour + β₄·DayOfWeek + ε

## Results

| Model | MSE | RMSE | Accuracy |
|---|---|---|---|
| PCA | N/A | N/A | 95% variance retained |
| FB Prophet | ~18.5 | ~4.3 | ~83% |
| ARIMA/SARIMA | ~21.2 | ~4.6 | ~81% |
| **Linear Regression** | **11.02** | **3.32** | **87.89%** |

## FB Prophet Forecast Results

images/FB_Prophet results.png

## Tech Stack
- Python · Pandas · NumPy · Scikit-learn · FB Prophet ·ARIMA/SARIMA (statsmodels) · Matplotlib · Raspberry Pi 4 ·PMS5003 sensor · DHT11 sensor · MQTT

## Key Findings

- Temperature and humidity are reliable linear predictors of outdoor PM2.5 in Karachi's urban environment
- PM2.5 highest in winter (Feb–Mar: 60–120 µg/m³), drops sharply in summer (Jun–Sep: 20–35 µg/m³)
- Saturday consistently shows highest PM2.5 (more vehicles/commerce)
- Morning rush hour (06:51–10:17) shows daily PM2.5 peak

## Future Work

- Incorporate additional sensors (wind speed, atmospheric pressure)
- Explore LSTM networks and Gradient Boosting on denser datasets
- Scale to multi-node city-wide air quality mapping
