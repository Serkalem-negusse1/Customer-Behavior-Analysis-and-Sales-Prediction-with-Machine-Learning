import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
from scripts.data_loader import load_data, preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sliding_window(data, window_size):
    """
    Create sliding window sequences for time-series data.
    
    Args:
        data (np.array): Time-series data.
        window_size (int): Size of the sliding window.
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values.
    """
    logging.info("Creating sliding window sequences.")
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def train_random_forest(train_df):
    """
    Train a RandomForestRegressor model.
    
    Args:
        train_df (pd.DataFrame): Preprocessed training data.
    
    Returns:
        Pipeline: Trained RandomForestRegressor pipeline.
    """
    logging.info("Training RandomForestRegressor model.")
    X = train_df.drop(columns=['Sales', 'Customers', 'Date'])
    y = train_df['Sales']

    for col in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']:
        X[col] = X[col].astype(str)

    categorical_cols = ['Weekday', 'MonthSegment', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    numeric_cols = ['CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear', 'DaysToHoliday', 'DaysAfterHoliday']

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mape = mean_absolute_percentage_error(y_val, y_pred)

    logging.info(f"RandomForestRegressor Mean Absolute Error (MAE): {mae}")
    logging.info(f"RandomForestRegressor Root Mean Squared Error (RMSE): {rmse}")
    logging.info(f"RandomForestRegressor Mean Absolute Percentage Error (MAPE): {mape}")

    # Save the RandomForestRegressor model
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    rf_model_filename = f"random_forest_model_{timestamp}.pkl"
    joblib.dump(pipeline, rf_model_filename)
    logging.info(f"RandomForestRegressor model saved as {rf_model_filename}")

    return pipeline

def train_lstm(train_df, store_id=1):
    """
    Train an LSTM model for a specific store.
    
    Args:
        train_df (pd.DataFrame): Preprocessed training data.
        store_id (int): Store ID to train the model for.
    
    Returns:
        tuple: (lstm_model, scaler, window_size, X_val, y_val, history)
    """
    logging.info(f"Training LSTM model for store {store_id}.")
    store_data = train_df[train_df['Store'] == store_id][['Date', 'Sales']]
    store_data = store_data.sort_values('Date')

    # Differencing to make the time series stationary
    store_data['SalesDiff'] = store_data['Sales'].diff().dropna()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_sales = scaler.fit_transform(store_data[['Sales']])

    # Create sliding window data
    window_size = 7
    X, y = create_sliding_window(scaled_sales, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into training and validation sets
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Hyperparameter tuning with Keras Tuner
    def build_model(hp):
        model = Sequential()
        model.add(Input(shape=(window_size, 1)))  # Define input shape explicitly
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            return_sequences=True
        ))
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            return_sequences=False
        ))
        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error'
        )
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='tuner_results',
        project_name='store_sales_lstm'
    )

    logging.info("Starting hyperparameter tuning.")
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=1)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Best Hyperparameters: {best_hps}")

    # Train the final model
    logging.info("Training the final LSTM model.")
    final_model = tuner.hypermodel.build(best_hps)
    history = final_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Save the LSTM model
    lstm_model_filename = f"lstm_model_store_{store_id}.pkl"
    joblib.dump(final_model, lstm_model_filename)
    logging.info(f"LSTM model saved as {lstm_model_filename}")

    # Save the scaler
    scaler_filename = f"scaler_store_{store_id}.pkl"
    joblib.dump(scaler, scaler_filename)
    logging.info(f"Scaler saved as {scaler_filename}")

    return final_model, scaler, window_size, X_val, y_val, history

def main():
    # Load and preprocess data
    train_path = load_data('E:/10Academy/Data04/train.csv')
    store_path = load_data('E:/10Academy/Data04/store.csv')
    test_path  = load_data('E:/10Academy/Data04/test.csv')
    
    train_df, test_df, store_df = load_data(train_path, test_path, store_path)
    train_df, test_df, scaler = preprocess_data(train_df, test_df, store_df)

    # Train RandomForestRegressor
    rf_pipeline = train_random_forest(train_df)

    # Train LSTM
    lstm_model, lstm_scaler, window_size = train_lstm(train_df)

if __name__ == "__main__":
    main()
