"""
Credit Card Fraud Detection System - Configuration
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DB_PATH = os.path.join(DATA_DIR, 'fraud_detection.db')

# Dataset URL (Kaggle Credit Card Fraud Detection)
DATASET_URL = "https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/download"
DATASET_NAME = "creditcard.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk thresholds
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.7

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'fraud-detection-secret-key-2024')
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
