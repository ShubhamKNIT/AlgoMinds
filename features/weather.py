import requests
import pandas as pd
import matplotlib.pyplot as plt
from config import OPEN_WEATHER_API_KEY
import base64
import io

def get_current_weather(city_name: str, country_code: str) -> dict:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},{country_code}&units=metric&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Weather API failed with status {response.status_code}")

    # print(f"Weather API Response: {response.json()}")  # Debugging line
    return response.json()

def get_forecast_weather(city_name: str, country_code: str, cnt: int = 7) -> dict:
    url = f"http://api.openweathermap.org/data/2.5/forecast/daily?q={city_name},{country_code}&units=metric&cnt={cnt}&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Weather Forecast API failed with status {response.status_code}")

    return response.json()

def clean_forecast_weather_data(raw_list: list) -> pd.DataFrame:
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

def clean_forecast_weather_data(raw: dict) -> pd.DataFrame:
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

def plot_temperature_data(df: pd.DataFrame, return_base64: bool = False) -> str | None:
    plt.figure(figsize=(12, 6))
    markers = ["o", "s", "^", "x", "d", "p"]
    columns = ["temp_day", "temp_min", "temp_max", "temp_night", "temp_eve", "temp_morn"]


    for col, marker in zip(columns, markers):
        if col in df.columns:
            plt.plot(df.index, df[col], marker=marker, label=col.replace("_", " ").title())

    plt.title("7-Day Weather Forecast: Temperature Trends")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{img_base64}"
    else:
        plt.show()

def plot_rain_data(df: pd.DataFrame, return_base64: bool = False) -> str | None:
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
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{img_base64}"
    else:
        plt.show()

def plot_wind_data(df: pd.DataFrame, return_base64: bool = False) -> str | None:
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
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{img_base64}"
    else:
        plt.show()
