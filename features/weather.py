import requests
import pandas as pd
import matplotlib.pyplot as plt
from config import OPEN_WEATHER_API_KEY

def fetch_weather_now(city_name, country_code):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},{country_code}&units=metric&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Weather API failed with status {response.status_code}")

    # print(f"Weather API Response: {response.json()}")  # Debugging line
    return response.json()

def fetch_weather_forecast(city_name, country_code, cnt=7):
    url = f"http://api.openweathermap.org/data/2.5/forecast/daily?q={city_name},{country_code}&units=metric&cnt={cnt}&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Weather Forecast API failed with status {response.status_code}")

    return response.json()

def clean_weather_forecast_data(raw_list):
    """Convert OneCall daily forecast list into a clean DataFrame."""
    records = []
    for entry in raw_list:
        record = {
            "date": pd.to_datetime(entry["dt"], unit="s").normalize(),
            "sunrise": pd.to_datetime(entry.get("sunrise"), unit="s"),
            "sunset": pd.to_datetime(entry.get("sunset"), unit="s"),

            # temps
            "temp_day": entry["temp"].get("day"),
            "temp_min": entry["temp"].get("min"),
            "temp_max": entry["temp"].get("max"),
            "temp_night": entry["temp"].get("night"),
            "temp_eve": entry["temp"].get("eve"),
            "temp_morn": entry["temp"].get("morn"),

            # feels_like
            "feels_day": entry["feels_like"].get("day"),
            "feels_night": entry["feels_like"].get("night"),
            "feels_eve": entry["feels_like"].get("eve"),
            "feels_morn": entry["feels_like"].get("morn"),

            # other
            "pressure": entry.get("pressure"),
            "humidity": entry.get("humidity"),
            "clouds": entry.get("clouds"),
            "pop": entry.get("pop"),
            "rain": entry.get("rain", 0),

            # wind
            "wind_speed": entry.get("speed"),
            "wind_deg": entry.get("deg"),
            "wind_gust": entry.get("gust"),

            # weather
            "weather_main": entry["weather"][0].get("main") if entry.get("weather") else None,
            "weather_desc": entry["weather"][0].get("description") if entry.get("weather") else None,
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    return df

def clean_weather_data(raw: dict) -> pd.DataFrame:
    record = {
        "city": raw.get("name"),
        "country": raw.get("sys", {}).get("country"),
        "dt": pd.to_datetime(raw.get("dt"), unit="s"),
        "timezone": raw.get("timezone"),

        # main metrics
        "temp": raw["main"].get("temp"),
        "feels_like": raw["main"].get("feels_like"),
        "temp_min": raw["main"].get("temp_min"),
        "temp_max": raw["main"].get("temp_max"),
        "pressure": raw["main"].get("pressure"),
        "humidity": raw["main"].get("humidity"),

        # wind
        "wind_speed": raw["wind"].get("speed"),
        "wind_deg": raw["wind"].get("deg"),
        "wind_gust": raw["wind"].get("gust"),

        # clouds/visibility
        "clouds": raw.get("clouds", {}).get("all"),
        "visibility": raw.get("visibility"),

        # weather desc
        "weather_main": raw["weather"][0].get("main") if raw.get("weather") else None,
        "weather_desc": raw["weather"][0].get("description") if raw.get("weather") else None,

        # sunrise/sunset
        "sunrise": pd.to_datetime(raw["sys"].get("sunrise"), unit="s"),
        "sunset": pd.to_datetime(raw["sys"].get("sunset"), unit="s"),
    }

    df = pd.DataFrame([record])
    df.set_index("dt", inplace=True)
    return df

def plot_weather_forecast(df):
    plt.figure(figsize=(12, 6))
    
    plt.plot(df.index, df["temp_day"], marker="o", label="Day Temp (°C)")
    plt.plot(df.index, df["temp_min"], marker="s", label="Min Temp (°C)")
    plt.plot(df.index, df["temp_max"], marker="^", label="Max Temp (°C)")
    plt.plot(df.index, df["temp_night"], marker="x", label="Night Temp (°C)")
    plt.plot(df.index, df["temp_eve"], marker="d", label="Evening Temp (°C)")
    plt.plot(df.index, df["temp_morn"], marker="p", label="Morning Temp (°C)")
    
    plt.title("7-Day Weather Forecast: Temperature Trends")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_rain_forecast(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

    # Rainfall bar plot
    axes[0].bar(df.index, df["rain"], alpha=0.6, color="blue", label="Rainfall (mm)")
    axes[0].set_ylabel("Rainfall (mm)")
    axes[0].set_title("Rainfall Forecast (7 Days)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Precipitation probability line plot
    axes[1].plot(df.index, df["pop"] * 100, color="red", marker="o", label="Precipitation Probability (%)")
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_title("Precipitation Probability Forecast (7 Days)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Rotate x-axis labels for both plots
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Rain & Precipitation Forecast (7 Days)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_wind_forecast(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["wind_speed"], marker="o", label="Wind Speed (m/s)")
    plt.plot(df.index, df["wind_gust"], marker="^", linestyle="--", label="Wind Gust (m/s)")
    
    plt.title("Wind Forecast (7 Days)")
    plt.xlabel("Date")
    plt.ylabel("Speed (m/s)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()