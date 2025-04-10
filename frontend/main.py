import streamlit as st
import pickle
import joblib
import sklearn
import os
import pandas as pd

st.title('AQI Prediction App')
st.write('This web app uses a machine learning model to\
          predict the Air Quality Index (AQI) of a city\
          based on the input features such as PM2.5, PM10,\
          NO, NO2, CO, O3, and SO2.')

encoder = pickle.load(open('./objects/encoder.pkl', 'rb'))
scaler = pickle.load(open('./objects/scaler.pkl', 'rb'))
feature_cols = pickle.load(open('./objects/feature_cols.pkl', 'rb'))
city_list = pickle.load(open('./objects/city_list.pkl', 'rb'))
model = pickle.load(open('./objects/model_no_3.pkl', 'rb'))

st.sidebar.title('Input Parameters')
st.sidebar.date_input('Select Date')
City = st.sidebar.selectbox('Select a city:', city_list)
PM2_5 = st.sidebar.number_input('PM2.5', min_value=0.0, max_value=500.0, value=25.0, step = 10.0)
PM10 = st.sidebar.number_input('PM10', min_value=0.0, max_value=500.0, value=25.0, step = 10.0)
NO = st.sidebar.number_input('NO', min_value=0.0, max_value=500.0, value=25.0, step = 10.0)
NO2 = st.sidebar.number_input('NO2', min_value=0.0, max_value=500.0, value=25.0, step = 10.0)
CO = st.sidebar.number_input('CO', min_value=0.0, max_value=40.0, value=0.5, step = 0.5)
O3 = st.sidebar.number_input('O3', min_value=0.0, max_value=1000.0, value=25.0, step = 10.0)
SO2 = st.sidebar.number_input('SO2', min_value=0.0, max_value=2000.0, value=25.0, step = 10.0)

input_data = {
    'City': City,
    'PM2.5': PM2_5,
    'PM10': PM10,
    'NO': NO,
    'NO2': NO2,
    'CO': CO,
    'O3': O3,
    'SO2': SO2
}

data = {}
for feature in feature_cols:
    base_feature = feature.split('_')[0] 
    if base_feature in input_data:
        data[feature] = input_data[base_feature]

city_encoded_dict = dict(zip(city_list, encoder.transform(city_list)))
data['City_Encoded'] = city_encoded_dict[City]

data_df = pd.DataFrame([data], columns=feature_cols)
data_scaled = scaler.transform(data_df)

# Create two columns for the UI
col1, col2 = st.columns([1, 1])

# Place the subheader in the first column
col1.subheader('Click on the Button to predict')

# Place the button in the second column
if col2.button('Predict'):
    prediction = model.predict(data_scaled)[0]
    st.subheader('AQI Prediction:')
    
    # Define color ranges for AQI
    color_ranges = {
        (0, 50): '#1FE140',
        (51, 100): '#F5B700',
        (101, 150): '#F26430',
        (151, 200): '#DF2935',
        (201, 300): '#D77A61',
        (301, float('inf')): '#4D5061'
    }
    
    # Find the appropriate color for the prediction
    prediction_color = 'black'
    for range_, color in color_ranges.items():
        if range_[0] <= prediction <= range_[1]:
            prediction_color = color
            break
    
    # Display the predicted AQI with the selected color in a colored box
    st.markdown(f'<div style="background-color:{prediction_color}; \
                padding:10px; border-radius:5px;"><p style="color:white;\
                 font-size:25px;">Predicted AQI: {prediction}</p></div>', unsafe_allow_html=True)




st.markdown('---')

# Load and display images in a single frame
image_folder = 'images'
image_files = sorted(os.listdir(image_folder))

# Choose an initial image index
st.subheader("Project Images")
image_index = st.slider("Slide Images", 0, len(image_files) - 1, 0)

# Display the selected image
image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split('.')[0].replace('_', ' ').title()
st.title(image_title)
st.image(image_path, use_container_width=True)

st.markdown('---')
st.subheader('Best Predictions')
st.write('Checkout this demo video to see the best predictions of the AQI using the models')

# Display the youtube video
st.video('https://youtu.be/7dp8cNXx9pU')

st.subheader('Project Link')
st.write('To know more about the project take a look on the Github repo link given below')
st.markdown('<a href="https://github.com/ShubhamKNIT/AlgoMinds">\
            <img src="https://github.com/fluidicon.png" alt="GitHub"\
            style="width: 50px; border-radius: 50%;"></a> AlgoMinds', unsafe_allow_html=True)
