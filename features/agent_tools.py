from air_quality import (
    fetch_air_now, 
    fetch_air_forecast, 
    clean_air_data, 
    plot_air_data
)
from weather import (
    fetch_weather_now, 
    fetch_weather_forecast, 
    clean_weather_data, 
    clean_weather_forecast_data,
    plot_weather_forecast, 
    plot_rain_forecast, 
    plot_wind_forecast
)
from predict_aqi import predict_aqi_df
from fetch_coordinates import get_city_coordinates
from langchain_core.tools import StructuredTool
import pandas as pd

# workflows.py

# -------------------------
# Air Quality Workflow
# -------------------------
def air_quality_workflow(city_name: str, country_code: str, cnt: int = None) -> dict:
    coords = get_city_coordinates(city_name=city_name, country_code=country_code)
    lat, lon = coords["lat"], coords["lon"]

    # Fetch and clean
    if cnt:  # forecast
        cnt = min(cnt, 7)
        raw_data = fetch_air_forecast(lat=lat, lon=lon, cnt=cnt)
    else:    # current
        raw_data = fetch_air_now(lat=lat, lon=lon)

    df = clean_air_data(raw_data)

    # Plot (already returns base64)
    plot_base64 = plot_air_data(df, return_base64=True)

    summary = f"Air quality data for {city_name} ({country_code}) retrieved successfully."
    return {"data": df, "plot": plot_base64, "summary": summary}

get_air_quality_tool = StructuredTool.from_function(
    name="air_quality_workflow",
    func=lambda city_name, country_code, cnt=None: air_quality_workflow(
        city_name=city_name, country_code=country_code, cnt=cnt
    ),
    description="Get air quality data (current or forecast) for a city and return plot (base64) and summary."
)

# -------------------------
# Weather Workflow
# -------------------------
def weather_workflow(city_name: str, country_code: str, cnt: int = None) -> dict:

    # Fetch and clean
    if cnt:
        cnt = min(cnt, 7)
        raw_data = fetch_weather_forecast(city_name=city_name, country_code=country_code, cnt=cnt)
        df = clean_weather_forecast_data(raw_data)
    else:
        raw_data = fetch_weather_now(city_name=city_name, country_code=country_code).get("list", [])
        df = clean_weather_data(raw_data)

    # Plots (already return base64)
    weather_plot = plot_weather_forecast(df, return_base64=True)
    rain_plot = plot_rain_forecast(df, return_base64=True)
    wind_plot = plot_wind_forecast(df, return_base64=True)

    summary = f"Weather data for {city_name} ({country_code}) retrieved successfully."
    return {
        "data": df,
        "plots": {"weather": weather_plot, "rain": rain_plot, "wind": wind_plot},
        "summary": summary
    }

get_weather_tool = StructuredTool.from_function(
    name="weather_workflow",
    func=lambda city_name, country_code, cnt=None: weather_workflow(
        city_name=city_name, country_code=country_code, cnt=cnt
    ),
    description="Get weather data (current or forecast) for a city and return plots (base64) and summary."
)

# -------------------------
# Predict AQI Workflow
# -------------------------
def predict_aqi_workflow(pm2_5: float, pm10: float, co: float, no2: float, so2: float, o3: float) -> dict:
    df = pd.DataFrame({"pm2.5": pm2_5, "pm10": pm10, "co": co, "no2": no2, "so2": so2, "o3": o3}, index=[0])
    predicted_df = predict_aqi_df(df)
    summary = f"Predicted AQI and Quality based on provided pollutants: {predicted_df['predicted_aqi'].iloc[-1]} ({predicted_df['aqi_quality'].iloc[-1]})"
    return {"data": predicted_df, "summary": summary}

predict_aqi_tool = StructuredTool.from_function(
    name="predict_aqi_workflow",
    func=lambda pm2_5, pm10, co, no2, so2, o3: predict_aqi_workflow(
        pm2_5=pm2_5, pm10=pm10, co=co, no2=no2, so2=so2, o3=o3
    ),
    description="Predict AQI from provided pollutant values and return summary."
)

# -------------------------
# Both Workflow (Air + Weather)
# -------------------------
def both_workflow(city_name: str, country_code: str, cnt: int = None) -> dict:
    aq_result = air_quality_workflow(city_name=city_name, country_code=country_code, cnt=cnt)
    weather_result = weather_workflow(city_name=city_name, country_code=country_code, cnt=cnt)
    summary = aq_result["summary"] + " " + weather_result["summary"]
    return {"air_quality": aq_result, "weather": weather_result, "summary": summary}

both_tool = StructuredTool.from_function(
    name="both_workflow",
    func=lambda city_name, country_code, cnt=None: both_workflow(
        city_name=city_name, country_code=country_code, cnt=cnt
    ),
    description="Fetch both air quality and weather data for a city and return combined results with plots and summaries."
)