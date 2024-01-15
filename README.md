<img width="439" alt="image" src="https://github.com/ManasChandan/ai_aqi_prediction/assets/61978958/320d4d51-7b99-4e13-adbf-54074fa14e2b">

https://ai-aqi-prediction.onrender.com/predict 

Sample Payload - 
{
    "date": "2023-10-15",
    "co": 667.57,
    "no": 2.82,
    "no2": 20.39,
    "o3": 113.01,
    "so2": 21.93,
    "pm2_5": 46.81,
    "pm10": 53.47,
    "nh3": 5.13
}

Developed a sophisticated Air Quality Index (AQI) prediction system. This project utilizes the OpenWeather API to fetch real-time weather data, employing a robust machine learning pipeline for daily model retraining and updates. The application seamlessly integrates with MongoDB for model storage, employing over 40 DecisionTreeRegressor and RandomForestRegressor models to ensure precise AQI predictions.

**Data Integration: ** Real-time weather data from OpenWeatherAPI is seamlessly integrated to enhance AQI predictions.
**Machine Learning Pipeline: **The project employs a dynamic machine learning pipeline for daily model updates, ensuring adaptability to changing environmental conditions.
**Model Evaluation:** Over 40 DecisionTreeRegressor and RandomForestRegressor models are rigorously tested, guaranteeing precision in AQI predictions.
**Database Integration:** MongoDB efficiently manages all data, including historical information, models, and predictions.
**FastAPI Endpoints: **User-friendly FastAPI endpoints provide effortless interaction, granting access to real-time AQI predictions and historical data.
**Containerization with Docker:** The entire application is containerized using Docker, facilitating easy deployment across diverse environments.
**Continuous Integration/Continuous Deployment (CI/CD):** Hosted on Render, the inbuilt CI/CD Pipeline of the render handling the ensuring a consistently high-performance level.
