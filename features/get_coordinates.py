import requests
from config import OPEN_WEATHER_API_KEY

def get_city_coordinates(city_name: str, country_code: str) -> dict:
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},{country_code}&limit=1&appid={OPEN_WEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Geocoding API failed with status {response.status_code}")
    
    data = response.json()
    if not data or "lat" not in data[0] or "lon" not in data[0]:
        raise Exception("No geocoding data returned from API")
    
    # print(f"Geocoding API Response: {data}")  # Debugging line
    return {
        "lat": data[0]["lat"],
        "lon": data[0]["lon"]
    }
