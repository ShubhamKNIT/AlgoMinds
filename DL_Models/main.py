import streamlit as st
import tensorflow as tf
import os

# Function to preprocess input data
def preprocess_input(PM2_5, PM10, NO2, CO, O3, SO2):
    # Create a tensor with shape (1, 9) for batch size 1 and 9 features
    input_data = tf.constant([[PM2_5, PM10, NO2, CO, O3, SO2]], dtype=tf.float32)
    return input_data


# Streamlit App
st.title('AQI Prediction App')

# Sidebar for Input Parameters
st.sidebar.title('Input Parameters')
st.sidebar.date_input('Select Date')
PM2_5 = st.sidebar.number_input('PM2.5', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
PM10 = st.sidebar.number_input('PM10', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
NO2 = st.sidebar.number_input('NO2', min_value=0.0, max_value=500.0, value=25.0, step=10.0)
CO = st.sidebar.number_input('CO', min_value=0.0, max_value=40.0, value=0.5, step=0.5)
O3 = st.sidebar.number_input('O3', min_value=0.0, max_value=1000.0, value=25.0, step=10.0)
SO2 = st.sidebar.number_input('SO2', min_value=0.0, max_value=2000.0, value=25.0, step=10.0)

# Preprocess input data
input_data = preprocess_input(PM2_5, PM10, NO2, CO, O3, SO2)

# Load the pre-trained models
model_names = ['model_91', 'model_169', 'model_187']
model_selection = st.selectbox('Select Model', model_names)

if model_selection == 'model_91':
    model_path = './DL_Models/models/model_91.h5'
elif model_selection == 'model_169':
    model_path = './DL_Models/models/model_169.h5'
elif model_selection == 'model_187':
    model_path = './DL_Models/models/model_187.h5'

model = tf.keras.models.load_model(model_path)

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)[0][0]
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

    aqi_quality_table = {
        (0, 50): 'Good',
        (51, 100): 'Satisfactory',
        (101, 150): 'Moderate',
        (151, 200): 'Poor',
        (201, 300): 'Very Poor',
        (301, float('inf')): 'Severe'
    }
    
    # Find the appropriate color for the prediction
    prediction_color = 'black'  # Default color if not found in color_ranges
    aqi_quality = ''
    for range_, color in color_ranges.items():
        if range_[0] <= prediction <= range_[1]:
            prediction_color = color
            aqi_quality = aqi_quality_table[range_]
            break
    
    # Display the predicted AQI with the selected color in a colored box
    st.markdown(f'<div style="background-color:{prediction_color}; padding:10px; border-radius:5px;">\
                    <p style="color:white; font-size:25px;">{aqi_quality}</p>\
                    <p style="color:white; font-size:25px;">Predicted AQI: {prediction}</p>\
                  </div>', unsafe_allow_html=True)

st.write('')
st.write('This web app uses a machine learning model to predict the Air Quality Index (AQI)\
          of a city based on the input features such as PM2.5, PM10, NO2, CO, O3, and SO2.')
st.write('')
st.write('Check out all the models, each model is named after the number of trainable parameters\
          in the model. Select the model and input the values of the features to get the AQI\
          prediction.')
st.write('More number of trainable parameters in the model means the model is more\
          complex and can capture more complex patterns in the data.')

st.markdown('---')

# Load and display images in a single frame
image_folder = './DL_Models/images'  # Path to your images folder
image_files = sorted(os.listdir(image_folder))  # Sort image files

# Choose an initial image index
st.subheader("Project Images")
image_index = st.slider("Slide Images", 0, len(image_files) - 1, 0)

# Display the selected image
image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split('.')[0].replace('_', ' ').title()
st.title(image_title)
st.image(image_path, use_column_width=True)


# Project Link
st.markdown('---')
st.subheader('Best Predictions')
st.write('Checkout this demo video to see the best predictions of the AQI using the models')

# Display the youtube video
st.video('https://youtu.be/5Rhdfst09Dc')

st.subheader('Project Link')
st.write('To know more about the project take a look on the Github repo link given below')
st.markdown('<a href="https://github.com/ShubhamKNIT/AlgoMinds">\
            <img src="https://github.com/fluidicon.png" alt="GitHub"\
            style="width: 50px; border-radius: 50%;"></a> AlgoMinds', unsafe_allow_html=True)
