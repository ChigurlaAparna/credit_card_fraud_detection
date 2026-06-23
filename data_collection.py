"""
Credit Card Fraud Detection - Data Collection and Exploration
Downloads the dataset and performs initial exploration
"""
import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# Configuration
DATA_DIR = 'data'
DATASET_PATH = os.path.join(DATA_DIR, 'creditcard.csv')

def download_dataset():
    """
    Download the Credit Card Fraud Detection dataset from Kaggle
    Using the raw GitHub version for reliable access
    """
    if os.path.exists(DATASET_PATH):
        print("Dataset already exists. Loading from file...")
        return load_dataset()
    
    print("Downloading Credit Card Fraud Detection dataset...")
    
    # Use the raw GitHub URL for the dataset
    url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    
    try:
        df = pd.read_csv(url)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Dataset downloaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error downloading: {e}")
        # Fallback: Create synthetic dataset if download fails
        print("Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset()

def create_synthetic_dataset():
    """
    Create a synthetic dataset with similar characteristics to real fraud data
    Used as fallback when download is not available
    """
    np.random.seed(42)
    n_samples = 284807  # Same as original dataset
    
    # Create Time feature (seconds elapsed between each transaction and the first transaction)
    time = np.random.exponential(scale=10000, size=n_samples)
    time = np.cumsum(time)
    time = (time - time.min()) / (time.max() - time.min()) * 172800  # 2 days in seconds
    
    # Create Amount feature (highly skewed, with outliers)
    amount_genuine = np.random.lognormal(mean=4.3, sigma=0.9, size=n_samples)
    amount_fraud = np.random.lognormal(mean=5.5, sigma=1.1, size=n_samples)
    
    # Create V1-V28 features (PCA components - normally distributed)
    v_features = np.random.randn(n_samples, 28)
    
    # Create Class (0 = genuine, 1 = fraud)
    # Approximately 0.172% fraud rate (similar to real data)
    fraud_rate = 0.00172
    is_fraud = np.random.binomial(1, fraud_rate, n_samples)
    
    # Adjust amounts based on fraud status
    amount = np.where(is_fraud == 1, amount_fraud, amount_genuine)
    
    # Create dataframe
    df = pd.DataFrame(v_features, columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = time
    df['Amount'] = amount
    df['Class'] = is_fraud
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Synthetic dataset created! Shape: {df.shape}")
    
    return df

def load_dataset():
    """Load the dataset from CSV file"""
    if not os.path.exists(DATASET_PATH):
        return download_dataset()
    return pd.read_csv(DATASET_PATH)

def explore_dataset(df):
    """
    Perform initial exploration of the dataset
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    print("\n📊 Basic Information:")
    print(f"   Total Samples: {len(df):,}")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n📋 Column Names:")
    print(f"   {list(df.columns)}")
    
    print("\n🔍 Data Types:")
    print(df.dtypes)
    
    print("\n📈 Statistical Summary:")
    print(df.describe().round(2))
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing[missing > 0])
    
    print("\n🎯 Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(f"   Genuine (0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"   Fraud (1):   {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")
    
    print("\n💰 Amount Statistics:")
    print(f"   Min: ${df['Amount'].min():.2f}")
    print(f"   Max: ${df['Amount'].max():.2f}")
    print(f"   Mean: ${df['Amount'].mean():.2f}")
    print(f"   Median: ${df['Amount'].median():.2f}")
    print(f"   Std: ${df['Amount'].std():.2f}")
    
    print("\n⏰ Time Statistics:")
    print(f"   Min: {df['Time'].min():.2f} seconds")
    print(f"   Max: {df['Time'].max():.2f} seconds")
    print(f"   Range: {(df['Time'].max() - df['Time'].min())/3600:.2f} hours")
    
    return df

if __name__ == "__main__":
    # Load and explore dataset
    df = download_dataset()
    explore_dataset(df)
    print("\n✅ Data collection and exploration complete!")
