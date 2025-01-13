## Customer Behavior Analysis and Sales Prediction with Machine Learning
## Overview
This project explores customer purchasing behavior and predicts store sales using advanced data analysis and machine learning techniques. It comprises three major tasks:

## Exploration of Customer Purchasing Behavior (EDA):

Analyzing customer behavior in response to factors like promotions, holidays, store openings, and assortment types.
Data cleaning, feature interaction analysis, and visualization to extract actionable insights.
Prediction of Store Sales:

Building predictive models using tree-based algorithms and deep learning (LSTM).
Preprocessing data, feature engineering, and evaluating models with custom loss functions.
Serialization of models for deployment and performance analysis.
Model Serving API:

Deploying the trained models via a REST API for real-time sales prediction.
Frameworks such as Flask or FastAPI are used for serving predictions.

# Key Features
Exploratory Data Analysis: Visualization and statistical insights into customer purchasing patterns.
Machine Learning Models: Regression-based prediction models, including random forests and LSTM.
API Development: Seamless integration of models into a production-ready API.



## How to Run the Code
# Clone this repository:
git clone <https://github.com/Serkalem-negusse1/Customer-Behavior-Analysis-and-Sales-Prediction-with-Machine-Learning>
cd <repository-folder>

# Install dependencies:
pip install -r requirements.txt

# Run the Jupyter notebooks for EDA and modeling:

Task 1: eda_customer_behavior.ipynb
Task 2: sales_prediction_pipeline.ipynb
Task 3: API scripts: app.py
Test the REST API locally:

python app.py

# Requirements
Python 3.8+
Key Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, TensorFlow/PyTorch, Flask/FastAPI.

# Contributor
Serkalem Negusse