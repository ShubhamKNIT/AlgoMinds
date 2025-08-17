import requests
from config import OPEN_WEATHER_API_KEY
from datetime import datetime, timezone
from predict_aqi import predict_aqi_df
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io

def get_current_air_quality(lat: float, lon: float) -> dict:
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"AQI API failed with status {response.status_code}")
    return response.json()

def get_forecast_air_quality(lat: float, lon: float, cnt: int = 7) -> dict:
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&cnt={cnt}&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"AQI Forecast API failed with status {response.status_code}")

    return response.json()

def clean_air_quality_data(data: dict, model_name: str = "model_91") -> pd.DataFrame:
    records = []
    for entry in data:
        record = {
            "datetime": datetime.fromtimestamp(entry["dt"].astype(int), tz=timezone.utc),
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
    df = pd.DataFrame(records)
    df.reset_index(drop=True, inplace=True)

    # Predict AQI if model is provided
    if model_name:
        predicted_df = predict_aqi_df(
            df[["pm2_5", "pm10", "no2", "co", "o3", "so2"]],
            model_name=model_name
        )
        df["aqi_predicted"] = predicted_df["aqi_predicted"]
        # df["aqi_quality"] = predicted_df["aqi_quality"]

    # attach units metadata
    df.attrs["units"] = {
        "CO": "µg/m³", "NO": "µg/m³", "NO2": "µg/m³", "O3": "µg/m³",
        "SO2": "µg/m³", "PM2_5": "µg/m³", "PM10": "µg/m³", "NH3": "µg/m³",
        "AQI_Index": "1–5 scale",
        "air_quality_predicted": "AQI value",
        "aqi_quality": "Good–Severe category"
    }

    return df

def plot_air_quality_data(df: pd.DataFrame, return_base64: bool = False) -> str | None:
    """Plot pollutant levels and predicted AQI, return as base64 or show inline."""
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left subplot: Pollutants ---
    pollutants = ["pm2_5", "pm10", "no2", "co", "o3", "so2"]
    for pollutant in pollutants:
        if pollutant in df.columns:
            axes[0].plot(df["datetime"], df[pollutant], label=pollutant.upper())
    
    axes[0].set_title("Pollutant Levels")
    axes[0].set_xlabel("Date/Time")
    axes[0].set_ylabel("Concentration (µg/m³)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # --- Right subplot: Predicted AQI ---
    axes[1].plot(
        df["datetime"], df["aqi_predicted"],
        color="red", label="Predicted AQI", linestyle="--", marker="o"
    )
    axes[1].set_title("Predicted AQI Over Time")
    axes[1].set_xlabel("Date/Time")
    axes[1].set_ylabel("AQI Value")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.tight_layout()

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
    else:
        plt.show()
        plt.close(fig)
        return None

# if __name__ == "__main__":
#     air_data = fetch_air_forecast(28.6139, 77.2090)  # Example coordinates for Delhi
#     cleaned_data = clean_air_data(air_data.get("list"), model_name="model_91")
#     print(cleaned_data)
#     cleaned_data = clean_air_data(air_data, model_name="model_91")
#     print(cleaned_data)

#     plot_air_data(cleaned_data)