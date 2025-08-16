import streamlit as st
from streamlit_folium import st_folium
import folium

# Import your modules
from features.fetch_coordinates import get_city_coordinates
from features.fetch_aqi import fetch_aqi_data
from features.fetch_weather import fetch_weather_data


# Initialize session state
if "city" not in st.session_state:
    st.session_state["city"] = None
if "lat" not in st.session_state:
    st.session_state["lat"] = None
if "lon" not in st.session_state:
    st.session_state["lon"] = None

st.title("AQI + Weather API Test UI")

# ---------------------
# City Input Section
# ---------------------
st.subheader("Option 1: Enter City Name")
city = st.text_input("City Name", "")

lat, lon = None, None
if city:
    try:
        lat, lon = get_city_coordinates(city)
        st.session_state["city"] = city
        st.session_state["lat"] = lat
        st.session_state["lon"] = lon
        st.success(f"✅ {city}: Latitude: {lat}, Longitude: {lon}")
        print(f"Coordinates for {city}: Latitude {lat}, Longitude {lon}")  # Debugging line
    except Exception as e:
        st.error(f"Geocoding Error: {e}")

# ---------------------
# Map Click Section
# ---------------------
st.subheader("Option 2: Select Location on Map")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
clicked = st_folium(m, width=700, height=500)

if clicked and clicked['last_clicked']:
    lat = clicked['last_clicked']['lat']
    lon = clicked['last_clicked']['lng']
    st.session_state["city"] = None  # override city if map clicked
    st.session_state["lat"] = lat
    st.session_state["lon"] = lon
    st.info(f"🗺️ Map Selected Coordinates: Latitude {lat}, Longitude {lon}")

# ---------------------
# Fetch Data Section
# ---------------------
if st.button("Fetch AQI & Weather") and st.session_state["lat"] is not None and st.session_state["lon"] is not None:
    try:
        # AQI Data
        aqi_data = fetch_aqi_data(st.session_state["lat"], st.session_state["lon"])
        st.subheader("Real-Time AQI & Pollutants")
        st.json(aqi_data)

        # Weather Data
        weather_data = fetch_weather_data(st.session_state["lat"], st.session_state["lon"])
        st.subheader("Detailed Weather Data")
        st.json(weather_data)

    except Exception as e:
        st.error(f"Error fetching data: {e}")

st.write("⚠️ You can either enter a city or click on the map. If both are provided, the map selection will override the city.")
