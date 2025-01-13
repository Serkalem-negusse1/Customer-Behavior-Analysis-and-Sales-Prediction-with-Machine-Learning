import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        logging.info(f"Loading data from {file_path}.")
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise

def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime format."""
    logging.info("Converting applicable columns to datetime format.")
    date_cols = ['Date']
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                logging.info(f"Converted {col} to datetime.")
            except Exception as e:
                logging.warning(f"Could not convert {col}: {e}")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    logging.info("Handling missing values.")
    fill_methods = {
        'CompetitionDistance': 'median',
        'CompetitionOpenSinceMonth': 'mode',
        'CompetitionOpenSinceYear': 'mode',
        'Promo2SinceWeek': 'mode',
        'Promo2SinceYear': 'mode',
        'PromoInterval': 'None'
    }

    for col, method in fill_methods.items():
        if col in df.columns:
            if method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(method, inplace=True)
            logging.info(f"Filled missing values in {col} using {method}.")
    return df

def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect missing values in the dataset and return a summary."""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    return pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})

def detect_outliers(df: pd.DataFrame, cols: list, threshold: float = 1.5) -> pd.DataFrame:
    """Detect outliers in specified columns using the IQR method."""
    logging.info("Detecting outliers.")
    outlier_summary = {}

    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_summary[col] = {
                'Lower Bound': lower,
                'Upper Bound': upper,
                'Outlier Count': len(outliers)
            }
            logging.info(f"Outliers detected in {col}: {len(outliers)}")

    return pd.DataFrame.from_dict(outlier_summary, orient='index')

def outlier_detection(df: pd.DataFrame, cols: list, threshold: float = 1.5) -> dict:
    """Detect outliers in specified columns using the IQR method and return details."""
    logging.info("Performing outlier detection.")
    outliers = {}
    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            logging.info(f"Detected outliers in {col}.")
    return outliers

def cap_outliers(df: pd.DataFrame, cols: list, threshold: float = 1.5) -> pd.DataFrame:
    """Cap outliers in specified columns to the IQR range."""
    logging.info("Capping outliers.")
    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
            logging.info(f"Capped outliers in {col}.")
    return df

def visualize_missing_data(df: pd.DataFrame, dataset_name: str):
    """Visualize missing values as a bar plot."""
    logging.info(f"Visualizing missing data for {dataset_name}.")
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_percent})
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if not missing_df.empty:
        missing_df.sort_values(by='Percentage', ascending=False).plot(
            kind='bar', y='Percentage', legend=False, color='skyblue', figsize=(10, 6))
        plt.title(f'Missing Data in {dataset_name}')
        plt.ylabel('Percentage of Missing Values')
        plt.xlabel('Columns')
        plt.tight_layout()
        plt.show()
    else:
        logging.info(f"No missing values in {dataset_name}.")

def visualize_outliers(df: pd.DataFrame, cols: list, dataset_name: str):
    """Visualize outliers in the specified columns using box plots."""
    logging.info(f"Visualizing outliers for {dataset_name}.")
    n_cols = 2
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i // n_cols, i % n_cols])
        axes[i // n_cols, i % n_cols].set_title(f'{col} Outliers')

    for j in range(len(cols), n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.tight_layout()
    plt.suptitle(f'Outlier Visualization for {dataset_name}', y=1.02)
    plt.show()

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, store_df: pd.DataFrame) -> tuple:
    """
    Preprocess the datasets including handling missing values, outliers, and feature engineering.
    
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        store_df (pd.DataFrame): Store data.
    
    Returns:
        tuple: (train_df, test_df, scaler)
    """
    # Handle missing values
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    store_df = handle_missing_values(store_df)
    
    # Handle outliers
    numeric_cols = ['CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear']
    train_df = cap_outliers(train_df, numeric_cols)
    test_df = cap_outliers(test_df, numeric_cols)
    store_df = cap_outliers(store_df, numeric_cols)
    
    # Feature extraction from 'Date' column
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    train_df['Weekday'] = train_df['Date'].dt.day_name()
    test_df['Weekday'] = test_df['Date'].dt.day_name()

    train_df['Weekend'] = train_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    test_df['Weekend'] = test_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

    train_df['MonthSegment'] = pd.cut(train_df['Date'].dt.day, bins=[0, 10, 20, 31], labels=['beginning', 'mid', 'end'])
    test_df['MonthSegment'] = pd.cut(test_df['Date'].dt.day, bins=[0, 10, 20, 31], labels=['beginning', 'mid', 'end'])

    # Merge datasets
    train_df = train_df.merge(store_df, on='Store', how='left')
    test_df = test_df.merge(store_df, on='Store', how='left')

    # Scale numeric features
    numeric_cols = ['CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear']
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, test_df, scaler

def preprocess_and_visualize(file_path: str, store_file_path: str):
    """Load, preprocess, and visualize the dataset."""
    data = load_data(file_path)
    store_data = load_data(store_file_path)
    
    data = preprocess_dates(data)
    data = handle_missing_values(data)
    store_data = handle_missing_values(store_data)

    visualize_missing_data(data, 'Sales Data')
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    visualize_outliers(data, numeric_cols, 'Sales Data')
    data = cap_outliers(data, numeric_cols)

    # Train-Test-Split and Preprocessing
    train_data, test_data, scaler = preprocess_data(data, data, store_data)

    logging.info("Preprocessing and visualization completed.")
    return train_data, test_data, scaler
