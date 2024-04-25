import streamlit as st
import pickle
import joblib
import sklearn
import os
import pandas as pd

# # Ignore warnings related to missing feature names
# warnings.filterwarnings("ignore", 
#                         category=UserWarning, 
#                         message="X does not have valid feature names, \
#                             but StandardScaler was fitted with feature names")


st.title('AQI Prediction App')
st.write('This web app uses a machine learning model to\
          predict the Air Quality Index (AQI) of a city\
          based on the input features PM2.5 and PM10 only.')

scaler = pickle.load(open('./exp/objects/scaler.pkl', 'rb'))
model = pickle.load(open('./exp/objects/models/model_no_2.pkl', 'rb'))


st.sidebar.title('Input Parameters')
PM2_5 = st.sidebar.number_input('PM2.5', min_value=0.0, max_value=500.0, value=20.0, step = 10.0)
PM10 = st.sidebar.number_input('PM10', min_value=0.0, max_value=500.0, value=20.0, step = 10.0)
feature_cols = ['PM2.5', 'PM10']


input_data = {
    'PM2.5': PM2_5,
    'PM10': PM10,
}

data_df = pd.DataFrame([input_data], columns=feature_cols)
data_scaled = scaler.transform(data_df)

col1, col2 = st.columns([1, 1])  # Divide the space into two columns

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
    prediction_color = 'black'  # Default color if not found in color_ranges
    for range_, color in color_ranges.items():
        if range_[0] <= prediction <= range_[1]:
            prediction_color = color
            break
    
    # Display the predicted AQI with the selected color in a colored box
    st.markdown(f'<div style="background-color:{prediction_color}; \
                padding:10px; border-radius:5px;"><p style="color:white;\
                 font-size:25px;">Predicted AQI: {prediction}</p></div>', unsafe_allow_html=True)




st.markdown('---')
# st.write('Input Data', input_data)
# st.write('Data', data)
# st.write('Scaled Data', data_scaled)


# Load and display images in a single frame
image_folder = './exp/images'  # Path to your images folder
image_files = sorted(os.listdir(image_folder))  # Sort image files

# Choose an initial image index
st.subheader("Project Images")
image_index = st.slider("Slide Images", 0, len(image_files) - 1, 0)

# Display the selected image
image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split('.')[0].replace('_', ' ').title()
st.title(image_title)
st.image(image_path, use_column_width=True)

st.markdown('---')

st.subheader('Project Link')
st.write('To know more about the project take a look on the Github repo link given below')
st.markdown('<a href="https://github.com/ShubhamKNIT/AlgoMinds">\
            <img src="https://github.com/fluidicon.png" alt="GitHub"\
            style="width: 50px; border-radius: 50%;"></a> AlgoMinds', unsafe_allow_html=True)
