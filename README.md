# AlgoMinds: AQI Prediction Project

This project focuses on predicting the Air Quality Index (AQI) using multiple machine learning and deep learning models, based on air pollutant data collected from Indian cities. The complete pipeline covers data cleaning, feature engineering, model training, evaluation, and deployment using Streamlit.

---

- **Demo Videos**:
  - [ScikitLearn Model](https://youtu.be/7dp8cNXx9pU)
  - [TensorFlow Model](https://youtu.be/5Rhdfst09Dc)

- **Live Models**:
  - [RandomForestRegressor Model](https://aqi-rfr.streamlit.app/)
  - [SGDRegressor Model](https://aqi-sgd.streamlit.app/)
  - [TensorFlow Model](https://aqi-nn.streamlit.app/)

The experimental model demonstrates superior performance compared to the main model. Explore these models through the provided links.

## Introduction

This project aims to develop machine learning models to predict the Air Quality Index (AQI) based on historical air quality data. The models are trained on the [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) dataset and evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShubhamKNIT/AlgoMinds
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Depending on the model you wish to interact with, run one of the following commands:

   - For the main model:
     ```bash
     streamlit run frontend/main.py
     ```

   - For the experimental model:
     ```bash
     streamlit run exp/main.py
     ```

   - For the TensorFlow model:
     ```bash
     streamlit run DL_Models/main.py
     ```

## Project Structure

The project directory is organized as follows:

- `data/`: Contains datasets used in the project.
  - `city_aqi_day.csv`: Raw dataset with daily AQI data.
  - `city_hour.csv.zip`: Raw dataset with hourly AQI data.
  - `clean_data.csv`: Cleaned dataset after preprocessing.
  - `no_missing.csv`: Dataset with missing values handled.

- `exp/`: Contains the experimental model which performs better than the main model.
  - `aqi_predictor_exp.ipynb`: Jupyter Notebook for the experimental model.
  - `images/`: Visualizations related to data exploration and model performance.
  - `main.py`: Streamlit application for the experimental model.
  - `objects/`: Serialized objects such as trained models and scalers.
    - `models/`: Contains trained model files (`model_no_1.pkl`, `model_no_2.pkl`, `model_no_3.pkl`).
    - `scaler.pkl`: Scaler object for data normalization.
  - `requirements.txt`: Dependencies for the experimental model.

- `DL_Models/`: Contains TensorFlow-based neural network models.
  - `aqi_predictor_nn_final.ipynb`: Jupyter Notebook for neural network models.
  - `images/`: Visualizations related to neural network architectures and performance.
    - `1_raw_data.png`: Visualization of raw data.
    - `2_clean_data.png`: Visualization of cleaned data.
    - `3_model_91.png`: Architecture of Model 91.
    - `4_model_169.png`: Architecture of Model 169.
    - `5_model_187.png`: Architecture of Model 187.
  - `main.py`: Streamlit application for neural network models.
  - `models/`: Saved neural network models in HDF5 format (`model_91.h5`, `model_169.h5`, `model_187.h5`).
  - `requirements.txt`: Dependencies for the neural network models.

- `frontend/`: Contains the main model's Streamlit application.
  - `main.py`: Streamlit application for the main model.

- `images/`: Contains various visualizations related to data exploration and model evaluation.
  - `a_pie.png`: Pie chart of AQI categories.
  - `b_city.png`: AQI distribution across cities.
  - `c_raw_dist.png`: Distribution of raw data.
  - `d_raw_pp.png`: Pair plot of raw data.
  - `e_raw_corr.png`: Correlation heatmap of raw data.
  - `f_clean_dist.png`: Distribution of cleaned data.
  - `g_clean_pp.png`: Pair plot of cleaned data.
  - `h_clean_corr.png`: Correlation heatmap of cleaned data.
  - `i_rfr_model.png`: Random Forest Regressor model performance.
  - `j_feature_importance.png`: Feature importance plot.
  - `k_test_shap.png`: SHAP values for test data.
  - `l_train_shap.png`: SHAP values for training data.

- `notebook/`: Contains Jupyter Notebooks for model development.
  - `aqi_predictor.ipynb`: Notebook for AQI prediction models.

- `objects/`: Serialized objects for the main model.
  - `city_list.pkl`: List of cities in the dataset.
  - `encoder.pkl`: Encoder object for categorical variables.
  - `feature_cols.pkl`: List of selected feature columns.
  - `model_no_1.pkl`: Trained model 1.
  - `model_no_2.pkl`: Trained model 2.
  - `model_no_3.pkl`: Trained model 3.
  - `scaler.pkl`: Scaler object for data normalization.

- `requirements.txt`: Project dependencies.


## 📁 Dataset

- **Source**: [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Features**:  
  `Date`, `City`, `PM2.5`, `PM10`, `NO`, `NO₂`, `NOx`, `NH₃`, `CO`, `SO₂`, `O₃`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI Label`

---

## 🧠 Models Overview

### ✅ **Model 1: Regression-Based AQI Prediction (Multi-feature Approach)**

**Objective**: Predict AQI using multiple pollutants via ensemble and linear models.

- **Cleaning & Preprocessing**:
  - Removed irrelevant columns: `NOx`, `NH₃`, `Benzene`, `Toluene`, `Xylene`
  - Imputed missing values via **three-step interpolation** grouped by `City`, `AQI Label`, and `Date`
  - Detected and replaced outliers with AQI-label-wise mean values
- **Feature Engineering**:
  - Selected 6 pollutants: `PM2.5`, `PM10`, `NO₂`, `CO`, `SO₂`, `O₃`
  - Computed **weekly rolling averages** (e.g., `PM2.5_rw_avg`)
  - Scaled features and split data (80% train / 20% test)
- **Models Tried**:
  - SGDRegressor, XGBoost, RandomForest, LGBM
- **Hyperparameter Tuning**:
  - Used RandomizedSearchCV with cross-validation
- **Final Models Benchmarked**:

| Model                 | R² Score | MSE      |
|----------------------|----------|----------|
| XGBoost              | 0.9327   | 780.43   |
| GradientBoosting     | 0.8349   | 1913.66  |
| **RandomForest** ✅   | **0.9490** | **591.35** |

📌 **Winner**: RandomForestRegressor (best balance of accuracy and efficiency)

---

### ✅ **Model 2: Lightweight AQI Prediction (PM2.5 & PM10 Focus)**

**Objective**: Build a simple yet effective model using just the two most critical features for AQI – PM2.5 and PM10.

- **Preprocessing**:
  - Missing values filled using AQI-label-wise interpolation
  - Enhanced outlier detection using **EllipticEnvelope** (contamination = 0.30)
- **Feature Selection**:
  - Only `PM2.5` and `PM10`
  - Dataset after cleaning: **52,000 instances** (70% train / 30% test)
- **Models Tried**:
  - ElasticNet, SGD, KNeighbors, ExtraTrees, Lasso, Lars, BayesianRidge, OMP
- **Final Model Evaluation**:

| Model         | MSE      |
|---------------|----------|
| ElasticNet    | 516.61   |
| KNeighbors    | **367.92** |
| **SGDRegressor** ✅ | 516.64   |

📌 **Winner**: **SGDRegressor** – selected based on overall performance and consistency through visual inspection

---

### ✅ **Model 3: Neural Network-Based AQI Prediction**

**Objective**: Leverage deep learning to capture non-linear relationships among pollutants.

- **Features Used**:  
  `PM2.5`, `PM10`, `NO₂`, `CO`, `SO₂`, `O₃`
- **Neural Network Architectures**:

| Model       | Layers                        | Params | Patience | Iterations | RMSE     |
|-------------|-------------------------------|--------|----------|------------|----------|
| Model_91    | [6 → 6 → 1]                   | 91     | 2        | 8          | 26.1955  |
| Model_169   | [12 → 6 → 1]                  | 169    | 2        | 24         | 25.8530  |
| **Model_187** ✅ | [12 → 6 → 3 → 1]             | 187    | 3        | 53         | **24.9424** |

- **Training**:
  - Loss: MSE, Optimizer: Adam (lr = 0.01), Metrics: RMSE & MSE
  - Early stopping used to avoid overfitting (patience: 2–3)

📌 **Winner**: **Model_187** – deeper architecture yielded best performance

---

## 🚀 Deployment

All selected models were deployed using **Streamlit** to enable real-time AQI prediction:

- **Deployed Models**:
  - `RandomForestRegressor` (Model 1)
  - `SGDRegressor` (Model 2)
  - `Model_91`, `Model_169`, and `Model_187` (Model 3)
- **Features**:
  - User inputs pollutant values for the respective model
  - Model returns **predicted AQI** dynamically
  - Clean UI with sliders, number fields, and model-specific prediction panels

---

## 📌 Final Takeaways

- Multiple modeling strategies were explored—traditional ML and deep learning—to predict AQI with varying complexity and accuracy.
- Outlier handling, thoughtful imputation, and context-aware feature engineering played a crucial role in boosting model performance.
- Streamlit deployment provides an accessible interface for end-users to interact and visualize model predictions in real time.
