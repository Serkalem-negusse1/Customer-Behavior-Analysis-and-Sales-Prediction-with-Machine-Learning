import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} with shape {data.shape}")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        raise

# Load datasets
try:
    train = load_data('E:/10Academy/Data04/train.csv')
    store = load_data('E:/10Academy/Data04/store.csv')
    test  = load_data('E:/10Academy/Data04/test.csv')
except Exception as e:
    print(f"Data loading failed: {e}")
    exit()


