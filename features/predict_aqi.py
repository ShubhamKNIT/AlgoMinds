import pandas as pd
import tensorflow as tf

def predict_aqi_df(df, model_name="model_91"):
    """
    Predict AQI for a DataFrame containing columns:
    ['pm2_5', 'pm10', 'no2', 'co', 'o3', 'so2']
    Returns the DataFrame with added 'predicted_aqi' and 'aqi_quality' columns.
    """
    # Load model
    model_paths = {
        "model_91": "./models/model_91.h5",
        "model_169": "./models/model_169.h5",
        "model_187": "./models/model_187.h5"
    }
    if model_name not in model_paths:
        raise ValueError(f"Unknown model: {model_name}")
    model = tf.keras.models.load_model(model_paths[model_name])

    # Preprocess input
    input_tensor = tf.constant([
        [row.pm2_5, row.pm10, row.no2, row.co / 1000, row.o3, row.so2]  # Convert CO ug/m3 → mg/m3
        for _, row in df.iterrows()
    ], dtype=tf.float32)

    # Predict AQI
    predictions = model.predict(input_tensor).flatten()

    # AQI categories
    # aqi_quality_table = {
    #     (0, 50): "Good",
    #     (51, 100): "Fair",
    #     (101, 150): "Moderate",
    #     (151, 200): "Poor",
    #     (201, 300): "Very Poor",
    #     (301, float("inf")): "Severe",
    # }
    # qualities = []
    # for p in predictions:
    #     for (low, high), label in aqi_quality_table.items():
    #         if low <= p <= high:
    #             qualities.append(label)
    #             break

    # Add to DataFrame
    df_ = df.copy(deep=True)
    df_['aqi_predicted'] = predictions
    # df_['aqi_quality'] = qualities

    return df_

# if __name__ == "__main__":
#     entry = [{"pm2_5": 35, "pm10": 50, "no2": 25, "co": 4000, "o3": 60, "so2": 10},
#              {"pm2_5": 40, "pm10": 55, "no2": 30, "co": 4500, "o3": 65, "so2": 15}]
#     df = pd.DataFrame(entry)
#     result = predict_aqi_df(df, "model_91")
#     print(result)
