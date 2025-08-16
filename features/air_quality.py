import requests
from config import OPEN_WEATHER_API_KEY
from datetime import datetime, timezone
from predict_aqi import predict_aqi_df
import matplotlib.pyplot as plt
import pandas as pd

def fetch_air_now(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"AQI API failed with status {response.status_code}")
    return response.json()

def fetch_air_forecast(lat, lon, cnt=7):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&cnt={cnt}&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"AQI Forecast API failed with status {response.status_code}")

    return response.json()

def clean_air_data(data, model_name=None):
    records = []
    for entry in data:
        record = {
            "datetime": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
            "aqi_index": entry["main"]["aqi"],  # renamed for clarity
            "pm2_5": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"],
            "no2": entry["components"]["no2"],
            "co": entry["components"]["co"],
            "o3": entry["components"]["o3"],
            "so2": entry["components"]["so2"],
        }
        records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records).set_index("datetime")

    # Predict AQI if model is provided
    if model_name:
        predicted_df = predict_aqi_df(
            df[["pm2_5", "pm10", "no2", "co", "o3", "so2"]],
            model_name=model_name
        )
        df["air_quality_predicted"] = predicted_df["predicted_aqi"]
        df["aqi_quality"] = predicted_df["aqi_quality"]

    # attach units metadata
    df.attrs["units"] = {
        "CO": "µg/m³", "NO": "µg/m³", "NO2": "µg/m³", "O3": "µg/m³",
        "SO2": "µg/m³", "PM2_5": "µg/m³", "PM10": "µg/m³", "NH3": "µg/m³",
        "AQI_Index": "1–5 scale",
        "air_quality_predicted": "AQI value",
        "aqi_quality": "Good–Severe category"
    }

    return df

def plot_air_data(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["PM2_5"], label="PM2.5", marker="o")
    plt.plot(df.index, df["PM10"], label="PM10", marker="s")
    plt.plot(df.index, df["NO2"], label="NO2", marker="^")
    plt.plot(df.index, df["CO"], label="CO", marker="x")
    plt.plot(df.index, df["O3"], label="O3", marker="d")
    plt.plot(df.index, df["SO2"], label="SO2", marker="p")

    plt.title("Air Quality Forecast (7 days)")
    plt.xlabel("Datetime")
    plt.ylabel("Concentration (µg/m³)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    air_data = fetch_air_forecast(28.6139, 77.2090)  # Example coordinates for Delhi
    cleaned_data = clean_air_data(air_data.get("list"), model_name="model_91")
    print(cleaned_data)
    # cleaned_data = clean_air_data(air_data, model_name="model_91")
    # print(cleaned_data)

    # plot_air_data(cleaned_data)