# AlgoMinds
Machine Learning Project to predict the Air Quality Index

- Demo Video 1: [Youtube Link](https://youtu.be/7dp8cNXx9pU)
- Demo Video 2: [Tensorflow Model](https://www.youtube.com/watch?v=5Rhdfst09Dc)

## Live Model
- Experimental Model: https://aqisgd-model.streamlit.app/
- Main Model: https://algominds-aqi-predictor.streamlit.app/
- Tensorflow Model: https://aqi-nn.streamlit.app/

Certainly experimental model performs better than the Main Model. Checkout these models from the link.

## Introduction/Overview
This project aims to develop machine learning models to predict AQI. The models are trained on [AQI Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) dataset and evaluated based on mean_squared_error.

## Getting Started
To get started with the project, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/ShubhamKNIT/AlgoMinds
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the code:
   ```
   streamlit run frontend/main.py
   ```
    or
   ```
   streamlit run exp/main.py
   ```
   or
   ```
   streamlit run DL_Models/main.py
   ```

## Project Structure
The project directory is structured as follows:
- `data/`: Contains dataset(s) used in the project.
- `models/`: Contains machine learning models implemented in the project.
- `images/`: Contains results obtained from model training and evaluation.
- `exp/`: Contains experimental model which perform certainly better than the model on the main branch
- `DL_Models`: Tensorflow trained AQI Models
