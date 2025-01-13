import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Seaborn style
sns.set_palette("Set2")


def plot_promotion_distribution(train, test):
    """
    Visualize the distribution of promotional activities in the training and test datasets.
    """
    logging.info("Generating promotion distribution plots for training and test datasets.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(data=train, x='Promo', ax=axes[0], palette='Set1')
    axes[0].set_title('Promotion Distribution in Training Set')
    axes[0].set_xlabel('Promotion Status')
    axes[0].set_ylabel('Count')

    sns.countplot(data=test, x='Promo', ax=axes[1], palette='Set2')
    axes[1].set_title('Promotion Distribution in Test Set')
    axes[1].set_xlabel('Promotion Status')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def analyze_sales_holidays(train):
    """
    Analyze and visualize sales trends during holiday and non-holiday periods.
    """
    logging.info("Analyzing sales data for holidays.")
    train['StateHoliday'] = train['StateHoliday'].fillna('0').astype(str)

    sales_holiday = train.groupby(['StateHoliday', 'SchoolHoliday'])['Sales'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=sales_holiday, x='StateHoliday', y='Sales', hue='SchoolHoliday', ax=axes[0])
    axes[0].set_title('Average Sales During Holidays')
    axes[0].set_xlabel('State Holiday Type')
    axes[0].set_ylabel('Average Sales')

    sns.boxplot(x='StateHoliday', y='Sales', data=train, ax=axes[1])
    axes[1].set_title('Sales Distribution During Holidays')
    axes[1].set_xlabel('State Holiday Type')
    axes[1].set_ylabel('Sales')

    plt.tight_layout()
    plt.show()


def plot_monthly_sales_trend(train):
    """
    Plot the average sales trend by month.
    """
    logging.info("Visualizing monthly sales trends.")
    train['Date'] = pd.to_datetime(train['Date'])
    train['Month'] = train['Date'].dt.month

    monthly_sales = train.groupby('Month')['Sales'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o', color='b')
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.grid(True)
    plt.show()


def plot_sales_vs_customers(train):
    """
    Plot the relationship between Sales and Customers with correlation analysis.
    """
    correlation = train['Sales'].corr(train['Customers'])
    logging.info(f"Correlation between Sales and Customers: {correlation:.2f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Customers', y='Sales', data=train, alpha=0.7, edgecolor='k')
    plt.title('Sales vs. Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.show()


def compare_promo_effect(train):
    """
    Compare the impact of promotions on sales and customer counts.
    """
    logging.info("Comparing the effect of promotions on sales and customers.")
    promo_effect = train.groupby('Promo').agg({'Sales': 'mean', 'Customers': 'mean'}).reset_index()

    promo_effect['SalesPerCustomer'] = promo_effect['Sales'] / promo_effect['Customers']
    logging.info(f"Promo Effect Analysis:\n{promo_effect}")

    promo_effect_melted = promo_effect.melt(id_vars='Promo', value_vars=['Sales', 'Customers', 'SalesPerCustomer'])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=promo_effect_melted, x='Promo', y='value', hue='variable', palette='Set2')
    plt.title('Impact of Promotions on Sales and Customers')
    plt.xlabel('Promotion Status')
    plt.ylabel('Value')
    plt.legend(title='Metric')
    plt.show()


def compare_assortment_sales(train, store):
    """
    Analyze sales performance by assortment type.
    """
    logging.info("Analyzing sales across assortment types.")
    train_store = pd.merge(train, store, on='Store', how='left')
    assortment_sales = train_store.groupby('Assortment')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=assortment_sales, x='Assortment', y='Sales', palette='viridis')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.show()


if __name__ == "__main__":
    # Load datasets
    train = pd.read_csv('train.csv')  # Replace with actual file path
    test = pd.read_csv('test.csv')  # Replace with actual file path
    store = pd.read_csv('store.csv')  # Replace with actual file path

    # Example function calls
    plot_promotion_distribution(train, test)
    analyze_sales_holidays(train)
    plot_monthly_sales_trend(train)
    plot_sales_vs_customers(train)
    compare_promo_effect(train)
    compare_assortment_sales(train, store)
