import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="PM2.5 Class Predictor")

PM25_CLASS_RANGES = {
    'Low': '0 - 35.4 µg/m³',
    'Moderate': '35.5 - 55.4 µg/m³',
    'Unhealthy': '55.5 - 150.4 µg/m³',
    'Very Unhealthy': '150.5 - 250.4 µg/m³',
    'Hazardous': '> 250.4 µg/m³'
}

future_data = pd.read_csv('./data/kathmandu_pm25_class_2025_4_to_2025_5_dataset.csv')

# --- Custom FeatureEngineer Class (Must be defined for loading the pipeline) ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to perform complex feature engineering steps:
    - Clipping numerical features to specified bounds.
    - Transforming wind direction into sine/cosine components and binary flags.
    - Creating an 'is_windy' flag and handling rare 'condition' categories.
    - Generating lagged features for key variables including the target.
    - Dropping the 'date' column and handling NaNs.
    """
    def __init__(self, clipping_bounds=None, rare_condition_threshold=100, lag_features=None, lags=None):
        # Define default clipping bounds
        self.clipping_bounds = clipping_bounds if clipping_bounds is not None else {
            'temperature': {'lower': 1.5, 'upper': 35.4},
            'pressure': {'lower': 855, 'upper': 885},
            'dew_point': {'lower': -10, 'upper': 25},
            'humidity': {'lower': 0, 'upper': 100}
        }
        self.rare_condition_threshold = rare_condition_threshold
        # Define default features for lagging
        self.lag_features = lag_features if lag_features is not None else \
                            ['temperature', 'humidity', 'wind_speed', 'dew_point', 'pressure']
        self.lags = lags if lags is not None else [1, 2, 3]
        # Mapping for wind directions to degrees
        self.direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
            'CALM': np.nan, 'VAR': np.nan
        }
        self.rare_conditions_ = None # To store rare conditions learned during fit

    def fit(self, X, y=None):
        # Ensure X is a DataFrame for proper column operations
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Learn rare conditions from the 'condition' column
        if 'condition' in X.columns:
            temp_condition_base = X['condition'].str.replace(' / Windy', '', regex=False)
            condition_counts = temp_condition_base.value_counts()
            self.rare_conditions_ = condition_counts[condition_counts < self.rare_condition_threshold].index
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # 1. Clipping numerical features according to predefined bounds
        for col, bounds in self.clipping_bounds.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].clip(lower=bounds['lower'], upper=bounds['upper'])

        # 2. Wind Transformation: Convert wind direction to numerical features
        if 'wind' in X_transformed.columns:
            X_transformed['wind_deg'] = X_transformed['wind'].map(self.direction_map)
            X_transformed['is_calm'] = (X_transformed['wind'] == 'CALM').astype(int)
            X_transformed['is_var'] = (X_transformed['wind'] == 'VAR').astype(int)
            # Fill NaN degrees (from 'CALM'/'VAR') with 0 for trigonometric encoding
            X_transformed['wind_deg'] = X_transformed['wind_deg'].fillna(0)
            X_transformed['wind_sin'] = np.sin(np.deg2rad(X_transformed['wind_deg']))
            X_transformed['wind_cos'] = np.cos(np.deg2rad(X_transformed['wind_deg']))
            # Drop the original 'wind' column and the intermediate 'wind_deg'
            X_transformed.drop(columns=['wind', 'wind_deg'], inplace=True)

        # 3. Condition Transformation: Create 'is_windy' and handle rare categories
        if 'condition' in X_transformed.columns:
            X_transformed['is_windy'] = X_transformed['condition'].str.contains('/ Windy', na=False).astype(int)
            X_transformed['condition_base'] = X_transformed['condition'].str.replace(' / Windy', '', regex=False)
            # Replace rare conditions with 'Other' based on what was learned during fit
            if self.rare_conditions_ is not None:
                X_transformed['condition'] = X_transformed['condition_base'].replace(self.rare_conditions_, 'Other')
            else:
                # Fallback if fit was not called or no rare conditions were identified
                X_transformed['condition'] = X_transformed['condition_base']
            X_transformed.drop(columns=['condition_base'], inplace=True)

        # 4. Add lagged features: Requires 'date' column for sorting and 'pm25_class' for lagging.
        if 'date' in X_transformed.columns:
            X_transformed.sort_values('date', inplace=True) # Sort by date for correct lagging
            # Lag numerical features
            for feature in self.lag_features:
                if feature in X_transformed.columns:
                    for lag in self.lags:
                        X_transformed[f'{feature}_lag{lag}'] = X_transformed[feature].shift(lag)
            # Lag the PM2.5 class (target) as a feature
            if 'pm25_class' in X_transformed.columns:
                for lag in self.lags:
                    X_transformed[f'pm25_class_lag{lag}'] = X_transformed['pm25_class'].shift(lag)

            X_transformed.drop(columns=['date'], inplace=True) # Drop date column after lagging

        # Drop rows with NaNs created by lagging (first few rows will have NaNs)
        # This is crucial for training data, but for single-row prediction,
        # we'll handle the NaNs from lagging by ensuring enough history is provided.
        X_transformed.dropna(inplace=True)

        # Reset index to ensure clean DataFrame for subsequent steps in the pipeline
        return X_transformed.reset_index(drop=True)


# --- Helper Function for Prediction with History ---
def predict_with_history(model, feature_engineer_instance, X_train_columns, history_buffer, new_data, max_lag=3):
    """
    Prepares input data for prediction by combining new data with historical data
    to correctly calculate lagged features, then makes a prediction.

    Args:
        model: The loaded trained scikit-learn pipeline (e.g., RandomForest or XGBoost).
        feature_engineer_instance: An instance of the FeatureEngineer class (fitted on training data).
        X_train_columns (list): A list of column names that the model was trained on.
        history_buffer (list of dict): A list of dictionaries, where each dict represents
                                      a historical day's data (including 'date' and 'pm25_class').
        new_data (dict): A dictionary containing the current day's input features.
        max_lag (int): The maximum lag used in your FeatureEngineer.

    Returns:
        str: The predicted PM2.5 class as a human-readable string.
    """
    # Create a copy of the history buffer and append the new data
    temp_history_df = pd.DataFrame(history_buffer + [new_data])
    temp_history_df['date'] = pd.to_datetime(temp_history_df['date'])

    # Apply feature engineering to the combined historical + new data
    # The FeatureEngineer will calculate lags based on this combined data
    engineered_data = feature_engineer_instance.transform(temp_history_df.copy())

    # Get the last row, which corresponds to the 'new_data' with calculated lags
    # This row will have valid lagged features because we provided history
    input_for_prediction = engineered_data.iloc[[-1]]

    # Ensure all columns from training data are present and in the correct order
    # Fill any missing columns (e.g., new one-hot encoded categories not seen in this row) with 0
    missing_cols = set(X_train_columns) - set(input_for_prediction.columns)
    for c in missing_cols:
        input_for_prediction[c] = 0
    input_for_prediction = input_for_prediction[X_train_columns]

    # Make prediction
    predicted_label = model.predict(input_for_prediction)[0]
    predicted_class = st.session_state.label_encoder.inverse_transform([predicted_label])[0]

    return predicted_class

# --- Load Models and Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the trained models, label encoder, and training columns."""
    try:
        rf_pipeline = joblib.load('./model_pipelines/random_forest_pipeline.pkl')
        xgb_pipeline = joblib.load('./model_pipelines/xgboost_pipeline.pkl')
        le = joblib.load('./model_pipelines/label_encoder.pkl')
        X_train_columns = joblib.load('./model_pipelines/X_train_columns.pkl')
        rf_accuracy = joblib.load('./model_pipelines/rf_validation_accuracy.pkl')
        xgb_accuracy = joblib.load('./model_pipelines/xgb_validation_accuracy.pkl')

        feature_engineer_instance = FeatureEngineer()

        return rf_pipeline, xgb_pipeline, le, X_train_columns, feature_engineer_instance, rf_accuracy, xgb_accuracy
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'random_forest_pipeline.pkl', "
                 "'xgboost_pipeline.pkl', 'label_encoder.pkl', and 'X_train_columns.pkl' "
                 "are in the same directory as this script.")
        st.stop()

rf_model, xgb_model, le_loaded, X_train_cols_loaded, fe_instance, rf_val_accuracy, xgb_val_accuracy = load_artifacts()
# Initialize session state for history buffer if not already present
if 'history_buffer' not in st.session_state:
    st.session_state.history_buffer = [
        {'date': '2025-04-26', 'pm25_class': le_loaded.transform(['Hazardous'])[0], 'temperature': 22.520833, 'dew_point': 7.000000, 'humidity': 36.583333, 'wind_speed': 6.770833, 'pressure': 862.937500, 'condition': 'Fair', 'wind': 'E', 'holiday': 2},
        {'date': '2025-04-27', 'pm25_class': le_loaded.transform(['Unhealthy'])[0], 'temperature': 12.091667, 'dew_point': 11.250000, 'humidity': 56.437500, 'wind_speed': 8.416667, 'pressure': 865.791667, 'condition': 'Fair', 'wind': 'E', 'holiday': 1},
        {'date': '2025-04-28', 'pm25_class': le_loaded.transform(['Low'])[0], 'temperature': 17.270833, 'dew_point': 12.812500, 'humidity': 77.395833, 'wind_speed': 6.458333, 'pressure': 865.187500, 'condition': 'Mostly Cloudy', 'wind': 'VAR', 'holiday': 0}
    ]
    st.session_state.label_encoder = le_loaded

# --- Streamlit UI ---
st.title("💨 PM2.5 Air Quality Class Predictor")
st.markdown("""
This application predicts the PM2.5 air quality class based on current meteorological conditions and recent historical data.
""")

st.header("Current Day's Meteorological Data")

# Input fields for current day's data
col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=40.0, value=19.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=73.6, step=0.1)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=7.4, step=0.1)
with col2:
    dew_point = st.number_input("Dew Point (°C)", min_value=-20.0, max_value=30.0, value=13.5, step=0.1)
    pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=864.9, step=0.1)
    wind_direction = st.selectbox("Wind Direction", options=['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'CALM', 'VAR'], index=16)
with col3:
    condition = st.selectbox("Weather Condition", options=['Fair', 'Mostly Cloudy', 'CALM'], index=1)
    holiday = st.selectbox("Is it a Holiday?", options=[0, 1, 2], index=0)
    current_date = st.date_input("Date for Prediction", value=pd.to_datetime('2025-04-29'))


# Prepare the new data point
new_data_point = {
    'date': current_date.strftime('%Y-%m-%d'), # Format date as string for consistency
    'temperature': temperature,
    'dew_point': dew_point,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'pressure': pressure,
    'condition': condition,
    'wind': wind_direction,
    'holiday': holiday
}

st.markdown("---")
st.header("Prediction")

# --- Display Model Accuracy in the Main App ---
st.subheader("Model Performance (Validation Set)")
st.markdown(f"**Random Forest Accuracy:** {rf_val_accuracy * 100:.2f}%")
st.markdown(f"**XGBoost Accuracy:** {xgb_val_accuracy * 100:.2f}%")
st.markdown("---")

if st.button("Predict PM2.5 Class"):
    # Add a temporary pm25_class to new_data_point for lagged feature calculation
    # This value will be shifted out, but it's needed by FeatureEngineer.
    # We can use a dummy value, as it's the future value we're predicting.
    new_data_point['pm25_class'] = st.session_state.history_buffer[-1]['pm25_class'] # Use last known class as dummy

    with st.spinner("Predicting..."):
        # Make prediction with Random Forest
        rf_prediction = predict_with_history(
            rf_model,
            fe_instance, # Pass the FeatureEngineer instance
            X_train_cols_loaded,
            st.session_state.history_buffer,
            new_data_point,
            max_lag=3 # Ensure this matches the lags in FeatureEngineer
        )

        # Make prediction with XGBoost
        xgb_prediction = predict_with_history(
            xgb_model,
            fe_instance, # Pass the FeatureEngineer instance
            X_train_cols_loaded,
            st.session_state.history_buffer,
            new_data_point,
            max_lag=3 # Ensure this matches the lags in FeatureEngineer
        )

    st.success("Prediction Complete!")
    st.write(f"**Random Forest Predicted PM2.5 Class:** {rf_prediction} (Range: {PM25_CLASS_RANGES.get(rf_prediction, 'N/A')})")
    st.write(f"**XGBoost Predicted PM2.5 Class:** {xgb_prediction} (Range: {PM25_CLASS_RANGES.get(xgb_prediction, 'N/A')})")

    # Update history buffer with the new data point for future predictions
    # Note: For a real app, you'd update this with the *actual* observed pm25_class
    # once it becomes available, not the dummy value or the prediction.
    # For demonstration, we'll append the new data point to the history.
    # If you want to use the *predicted* class for future lags, you'd do:
    # st.session_state.history_buffer.append({**new_data_point, 'pm25_class': st.session_state.label_encoder.transform([rf_prediction])[0]})
    # For now, let's just append the new data point as is, without the predicted class
    # for its own pm25_class value, as it's a future prediction.
    # A robust system would ingest actual observed data to update history.
    st.session_state.history_buffer.append(new_data_point)
    # Ensure history buffer doesn't grow indefinitely
    if len(st.session_state.history_buffer) > 3: # max_lag
        st.session_state.history_buffer.pop(0)

st.markdown("---")
st.subheader("Current History Buffer (Last 3 Days)")
history_df_display = pd.DataFrame(st.session_state.history_buffer)
history_df_display['pm25_class_label'] = st.session_state.label_encoder.inverse_transform(history_df_display['pm25_class'])
st.dataframe(history_df_display[['date', 'temperature', 'humidity', 'pm25_class_label']])

st.markdown("""
**Note on History Buffer:** For a production application, the `history_buffer` would typically be loaded from a persistent data store (like a database)
and updated with actual observed data, not just the input or predicted values.
The `pm25_class` in the `new_data_point` is a placeholder that gets shifted out;
for accurate future predictions, the `history_buffer` needs to contain the true past `pm25_class` values.
""")