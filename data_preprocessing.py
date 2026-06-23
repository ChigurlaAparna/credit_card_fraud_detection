"""
Credit Card Fraud Detection - Data Preprocessing Pipeline
Handles missing values, scaling, SMOTE, and train-test split
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime

DATA_DIR = 'data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def handle_missing_values(df):
    """Check and handle missing values"""
    print("\n🔍 Checking for missing values...")
    
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing == 0:
        print("   ✅ No missing values found!")
        return df
    
    print(f"   ⚠️ Found {total_missing} missing values")
    
    # Handle missing values based on data type
    for col in df.columns:
        if missing[col] > 0:
            if df[col].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                df[col].fillna(df[col].median(), inplace=True)
                print(f"   Filled {col} with median")
            else:
                # Fill categorical columns with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"   Filled {col} with mode")
    
    return df

def scale_features(df, scaler=None, fit=True):
    """Scale numerical features"""
    print("\n⚙️ Scaling features...")
    
    # Features to scale (exclude Time and Class)
    scale_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    
    if fit:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
        print(f"   ✅ Fitted and transformed {len(scale_cols)} features")
    else:
        df_scaled = df.copy()
        df_scaled[scale_cols] = scaler.transform(df[scale_cols])
        print(f"   ✅ Transformed {len(scale_cols)} features")
    
    return df_scaled, scaler

def split_data(df, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print("\n📊 Splitting data...")
    
    # Separate features and target
    X = df.drop(['Class', 'Time'], axis=1)  # Remove Time as it's not useful for prediction
    y = df['Class']
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Testing set: {len(X_test):,} samples")
    print(f"   Training fraud rate: {y_train.mean()*100:.3f}%")
    print(f"   Testing fraud rate: {y_test.mean()*100:.3f}%")
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to handle class imbalance"""
    print("\n⚖️ Applying SMOTE for class imbalance...")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    original_counts = y_train.value_counts()
    resampled_counts = pd.Series(y_resampled).value_counts()
    
    print(f"   Original: Genuine={original_counts[0]:,}, Fraud={original_counts[1]:,}")
    print(f"   After SMOTE: Genuine={resampled_counts[0]:,}, Fraud={resampled_counts[1]:,}")
    print(f"   ⚖️ Class balance achieved!")
    
    return X_resampled, y_resampled, smote

def preprocess_pipeline(df, test_size=0.2, random_state=42, apply_smote_flag=True):
    """
    Complete preprocessing pipeline
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df.copy())
    
    # Step 2: Scale features
    df_scaled, scaler = scale_features(df_clean, fit=True)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=test_size, random_state=random_state
    )
    
    # Step 4: Apply SMOTE (only to training data)
    if apply_smote_flag:
        X_train_balanced, y_train_balanced, smote = apply_smote(X_train, y_train, random_state)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        smote = None
    
    # Save preprocessors
    print("\n💾 Saving preprocessors...")
    preprocessors = {
        'scaler': scaler,
        'smote': smote,
        'feature_columns': list(X_train.columns),
        'random_state': random_state,
        'test_size': test_size
    }
    
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessors.joblib')
    joblib.dump(preprocessors, preprocessor_path)
    print(f"   ✅ Preprocessors saved to: {preprocessor_path}")
    
    # Save processed datasets
    print("\n💾 Saving processed datasets...")
    
    # Save balanced training data
    train_balanced = pd.concat([X_train_balanced, y_train_balanced.reset_index(drop=True)], axis=1)
    train_balanced.to_csv(os.path.join(PROCESSED_DIR, 'train_balanced.csv'), index=False)
    
    # Save original training data (without SMOTE) for comparison
    train_original = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    train_original.to_csv(os.path.join(PROCESSED_DIR, 'train_original.csv'), index=False)
    
    # Save test data
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    test_data.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    
    print(f"   ✅ Processed datasets saved to: {PROCESSED_DIR}/")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return {
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'smote': smote,
        'feature_columns': list(X_train.columns)
    }

def load_preprocessed_data():
    """Load preprocessed datasets"""
    print("\n📂 Loading preprocessed data...")
    
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_balanced.csv'))
    y_train = X_train.pop('Class')
    
    test = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
    X_test = test.drop('Class', axis=1)
    y_test = test['Class']
    
    preprocessors = joblib.load(os.path.join(PROCESSED_DIR, 'preprocessors.joblib'))
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessors': preprocessors
    }

if __name__ == "__main__":
    from data_collection import load_dataset
    
    # Load and preprocess
    df = load_dataset()
    result = preprocess_pipeline(df)
    
    print("\n✅ Preprocessing complete!")
    print(f"\nDataset shapes:")
    print(f"   X_train: {result['X_train'].shape}")
    print(f"   X_test: {result['X_test'].shape}")
