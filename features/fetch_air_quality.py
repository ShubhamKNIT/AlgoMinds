import requests
from features.config import OPEN_WEATHER_API_KEY
from datetime import datetime
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

def clean_air_data(data):
    records = []
    for entry in data:
        record = {
            "datetime": datetime.fromtimestamp(entry["dt"]),
            "aqi_index": entry["main"]["aqi"],   # renamed for clarity
            "co": entry["components"]["co"],
            "no": entry["components"]["no"],
            "no2": entry["components"]["no2"],
            "o3": entry["components"]["o3"],
            "so2": entry["components"]["so2"],
            "pm2_5": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"],
            "nh3": entry["components"]["nh3"],
        }
        records.append(record)

    df = pd.DataFrame(records).set_index("datetime")
    
    # attach units metadata
    df.attrs["units"] = {
        "co": "µg/m³", "no": "µg/m³", "no2": "µg/m³", "o3": "µg/m³",
        "so2": "µg/m³", "pm2_5": "µg/m³", "pm10": "µg/m³", "nh3": "µg/m³",
        "aqi_index": "1–5 scale"
    }
    
    return df

def plot_air_data(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["pm2_5"], label="PM2.5", marker="o")
    plt.plot(df.index, df["pm10"], label="PM10", marker="s")
    plt.plot(df.index, df["no2"], label="NO2", marker="^")
    plt.plot(df.index, df["co"], label="CO", marker="x")
    plt.plot(df.index, df["o3"], label="O3", marker="d")
    plt.plot(df.index, df["so2"], label="SO2", marker="p")

    plt.title("Air Quality Forecast (7 days)")
    plt.xlabel("Datetime")
    plt.ylabel("Concentration (µg/m³)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()