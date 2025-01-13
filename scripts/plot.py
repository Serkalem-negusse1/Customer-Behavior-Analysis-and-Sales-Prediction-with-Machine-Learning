import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def plot_sales_over_time(store_data):
    """
    Plot sales over time for a specific store.
    
    Args:
        store_data (pd.DataFrame): DataFrame containing 'Date' and 'Sales' columns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(store_data['Date'], store_data['Sales'], label='Sales', color='blue')
    plt.title('Sales Over Time for Store 1')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_differenced_sales(store_data):
    """
    Plot differenced sales over time for a specific store.
    
    Args:
        store_data (pd.DataFrame): DataFrame containing 'Date' and 'SalesDiff' columns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(store_data['Date'].iloc[1:], store_data['SalesDiff'].iloc[1:], label='Differenced Sales', color='orange')
    plt.title('Differenced Sales Over Time for Store 1')
    plt.xlabel('Date')
    plt.ylabel('Differenced Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_training_validation_loss(history):
    """
    Plot the training and validation loss from the LSTM model training process.
    
    Args:
        history (keras.callbacks.History): Training history containing loss and validation loss.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred):
    """
    Plot actual vs predicted sales values.
    
    Args:
        y_true (np.array): Actual sales values.
        y_pred (np.array): Predicted sales values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Sales', color='blue')
    plt.plot(y_pred, label='Predicted Sales', color='red')
    plt.title('Actual vs Predicted Sales (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_future_predictions(future_dates, future_predictions):
    """
    Plot future sales predictions.
    
    Args:
        future_dates (pd.DatetimeIndex): Dates for future predictions.
        future_predictions (np.array): Predicted sales values for future dates.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_predictions, label='Predicted Sales', color='green', marker='o')
    plt.title('Sales Predictions for Future Dates')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()