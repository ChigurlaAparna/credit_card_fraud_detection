"""
Credit Card Fraud Detection - Feature Engineering
Creates meaningful fraud-related features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

DATA_DIR = 'data'
FEATURED_DIR = os.path.join(DATA_DIR, 'featured')
os.makedirs(FEATURED_DIR, exist_ok=True)

def create_transaction_features(df):
    """
    Create fraud-related features based on transaction patterns
    """
    print("\n🔧 Creating engineered features...")
    
    df_feat = df.copy()
    
    # 1. Amount-based features
    print("   Creating amount-based features...")
    df_feat['Amount_Log'] = np.log1p(df_feat['Amount'])
    df_feat['Amount_Squared'] = df_feat['Amount'] ** 2
    df_feat['Amount_Binned'] = pd.cut(df_feat['Amount'], 
                                       bins=[0, 10, 50, 100, 200, 500, 1000, float('inf')],
                                       labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    
    # 2. Time-based features
    print("   Creating time-based features...")
    df_feat['Hour'] = (df_feat['Time'] / 3600) % 24
    df_feat['Is_Night'] = ((df_feat['Hour'] >= 22) | (df_feat['Hour'] <= 6)).astype(int)
    df_feat['Is_Business_Hours'] = ((df_feat['Hour'] >= 9) & (df_feat['Hour'] <= 17)).astype(int)
    
    # Time periods
    df_feat['Time_Period'] = pd.cut(df_feat['Hour'],
                                    bins=[0, 6, 12, 18, 24],
                                    labels=[0, 1, 2, 3]).astype(int)
    
    # 3. V-features aggregations
    print("   Creating V-feature aggregations...")
    v_features = [f'V{i}' for i in range(1, 29)]
    
    # Mean, std, sum, min, max of V features
    df_feat['V_Mean'] = df_feat[v_features].mean(axis=1)
    df_feat['V_Std'] = df_feat[v_features].std(axis=1)
    df_feat['V_Sum'] = df_feat[v_features].sum(axis=1)
    df_feat['V_Min'] = df_feat[v_features].min(axis=1)
    df_feat['V_Max'] = df_feat[v_features].max(axis=1)
    df_feat['V_Range'] = df_feat['V_Max'] - df_feat['V_Min']
    
    # Count of extreme values (potential fraud indicators)
    df_feat['V_Negative_Count'] = (df_feat[v_features] < 0).sum(axis=1)
    df_feat['V_Positive_Count'] = (df_feat[v_features] > 0).sum(axis=1)
    
    # 4. Interaction features between top fraud-correlated features
    print("   Creating interaction features...")
    top_fraud_features = ['V14', 'V17', 'V12', 'V10', 'V4']
    
    # Products of top features
    df_feat['V14_V17'] = df_feat['V14'] * df_feat['V17']
    df_feat['V12_V14'] = df_feat['V12'] * df_feat['V14']
    df_feat['V10_V12'] = df_feat['V10'] * df_feat['V12']
    df_feat['V4_V14'] = df_feat['V4'] * df_feat['V14']
    
    # Ratios (with small epsilon to avoid division by zero)
    epsilon = 1e-6
    df_feat['V14_V17_Ratio'] = df_feat['V14'] / (df_feat['V17'] + epsilon)
    df_feat['V12_V10_Ratio'] = df_feat['V12'] / (df_feat['V10'] + epsilon)
    
    # 5. Distance from mean features
    print("   Creating distance features...")
    for feat in top_fraud_features:
        mean_val = df_feat[feat].mean()
        std_val = df_feat[feat].std()
        df_feat[f'{feat}_ZScore'] = (df_feat[feat] - mean_val) / std_val
        df_feat[f'{feat}_Dist_From_Mean'] = abs(df_feat[feat] - mean_val)
    
    # 6. Aggregate risk score
    df_feat['Risk_Score'] = (
        df_feat['V14_Dist_From_Mean'].clip(0, 3) +
        df_feat['V17_Dist_From_Mean'].clip(0, 3) +
        df_feat['V12_Dist_From_Mean'].clip(0, 3) +
        df_feat['V10_Dist_From_Mean'].clip(0, 3) +
        df_feat['V4_Dist_From_Mean'].clip(0, 3)
    )
    
    # 7. Polynomial features for top V features
    for feat in ['V14', 'V17', 'V12']:
        df_feat[f'{feat}_Squared'] = df_feat[feat] ** 2
        df_feat[f'{feat}_Cubed'] = df_feat[feat] ** 3
    
    print(f"   ✅ Created {len(df_feat.columns) - len(df.columns)} new features")
    print(f"   Total features: {len(df_feat.columns)}")
    
    return df_feat

def create_risk_indicators(df):
    """
    Create specific risk indicator features
    """
    print("\n🎯 Creating risk indicators...")
    
    df_risk = df.copy()
    
    # High amount flag
    amount_99 = df_risk['Amount'].quantile(0.99)
    df_risk['Is_High_Amount'] = (df_risk['Amount'] > amount_99).astype(int)
    
    # Extreme V14 flag (typically most predictive)
    v14_05 = df_risk['V14'].quantile(0.05)
    v14_95 = df_risk['V14'].quantile(0.95)
    df_risk['V14_Extreme'] = ((df_risk['V14'] < v14_05) | (df_risk['V14'] > v14_95)).astype(int)
    
    # Combined extreme flags
    df_risk['Multiple_Extremes'] = (
        (df_risk['V14_Extreme'].astype(int) + 
         df_risk['V17'].clip(-3, 3).abs().round() +
         df_risk['V12'].clip(-3, 3).abs().round()) > 2
    ).astype(int)
    
    # Time + Amount combination
    df_risk['Night_High_Amount'] = (df_risk['Is_Night'] & df_risk['Is_High_Amount']).astype(int)
    
    # V-feature anomaly count
    v_features = [f'V{i}' for i in range(1, 29)]
    df_risk['Anomaly_Count'] = 0
    for feat in v_features:
        q01, q99 = df_risk[feat].quantile(0.01), df_risk[feat].quantile(0.99)
        df_risk['Anomaly_Count'] += ((df_risk[feat] < q01) | (df_risk[feat] > q99)).astype(int)
    
    print(f"   ✅ Risk indicators created")
    
    return df_risk

def engineer_features(df):
    """
    Complete feature engineering pipeline
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Create all features
    df_feat = create_transaction_features(df)
    df_risk = create_risk_indicators(df_feat)
    
    # Select final features (exclude original Time for prediction)
    feature_cols = [col for col in df_risk.columns if col not in ['Class', 'Time']]
    
    print(f"\n📊 Final feature count: {len(feature_cols)}")
    print(f"   Feature list: {feature_cols}")
    
    # Save feature summary
    summary = {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'new_features': [col for col in feature_cols if col not in df.columns]
    }
    
    # Save featured dataset
    featured_path = os.path.join(FEATURED_DIR, 'featured_data.csv')
    df_risk.to_csv(featured_path, index=False)
    print(f"\n💾 Featured dataset saved to: {featured_path}")
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    
    return df_risk, feature_cols, summary

def get_feature_importance_names():
    """
    Return meaningful names for features (for visualization)
    """
    feature_names = {
        'V1': 'V1 (Transaction Pattern 1)',
        'V2': 'V2 (Transaction Pattern 2)',
        'V3': 'V3 (Transaction Pattern 3)',
        'V4': 'V4 (Transaction Pattern 4)',
        'V5': 'V5 (Transaction Pattern 5)',
        'V6': 'V6 (Transaction Pattern 6)',
        'V7': 'V7 (Transaction Pattern 7)',
        'V8': 'V8 (Transaction Pattern 8)',
        'V9': 'V9 (Transaction Pattern 9)',
        'V10': 'V10 (Transaction Pattern 10)',
        'V11': 'V11 (Transaction Pattern 11)',
        'V12': 'V12 (Transaction Pattern 12)',
        'V13': 'V13 (Transaction Pattern 13)',
        'V14': 'V14 (Transaction Pattern 14)',
        'V15': 'V15 (Transaction Pattern 15)',
        'V16': 'V16 (Transaction Pattern 16)',
        'V17': 'V17 (Transaction Pattern 17)',
        'V18': 'V18 (Transaction Pattern 18)',
        'V19': 'V19 (Transaction Pattern 19)',
        'V20': 'V20 (Transaction Pattern 20)',
        'V21': 'V21 (Transaction Pattern 21)',
        'V22': 'V22 (Transaction Pattern 22)',
        'V23': 'V23 (Transaction Pattern 23)',
        'V24': 'V24 (Transaction Pattern 24)',
        'V25': 'V25 (Transaction Pattern 25)',
        'V26': 'V26 (Transaction Pattern 26)',
        'V27': 'V27 (Transaction Pattern 27)',
        'V28': 'V28 (Transaction Pattern 28)',
        'Amount': 'Transaction Amount',
        'Amount_Log': 'Log-Transformed Amount',
        'Amount_Squared': 'Amount Squared',
        'Amount_Binned': 'Amount Category',
        'Hour': 'Hour of Day',
        'Is_Night': 'Night Transaction',
        'Is_Business_Hours': 'Business Hours',
        'Time_Period': 'Time Period',
        'V_Mean': 'Mean of V Features',
        'V_Std': 'Std of V Features',
        'V_Sum': 'Sum of V Features',
        'V_Min': 'Min of V Features',
        'V_Max': 'Max of V Features',
        'V_Range': 'Range of V Features',
        'V_Negative_Count': 'Negative V Features Count',
        'V_Positive_Count': 'Positive V Features Count',
        'V14_V17': 'V14 × V17 Interaction',
        'V12_V14': 'V12 × V14 Interaction',
        'V10_V12': 'V10 × V12 Interaction',
        'V4_V14': 'V4 × V14 Interaction',
        'V14_V17_Ratio': 'V14 / V17 Ratio',
        'V12_V10_Ratio': 'V12 / V10 Ratio',
        'V14_ZScore': 'V14 Z-Score',
        'V17_ZScore': 'V17 Z-Score',
        'V12_ZScore': 'V12 Z-Score',
        'V10_ZScore': 'V10 Z-Score',
        'V4_ZScore': 'V4 Z-Score',
        'V14_Dist_From_Mean': 'V14 Distance from Mean',
        'V17_Dist_From_Mean': 'V17 Distance from Mean',
        'V12_Dist_From_Mean': 'V12 Distance from Mean',
        'V10_Dist_From_Mean': 'V10 Distance from Mean',
        'V4_Dist_From_Mean': 'V4 Distance from Mean',
        'Risk_Score': 'Aggregated Risk Score',
        'V14_Squared': 'V14 Squared',
        'V17_Squared': 'V17 Squared',
        'V12_Squared': 'V12 Squared',
        'V14_Cubed': 'V14 Cubed',
        'V17_Cubed': 'V17 Cubed',
        'V12_Cubed': 'V12 Cubed',
        'Is_High_Amount': 'High Amount Flag',
        'V14_Extreme': 'V14 Extreme Value',
        'Multiple_Extremes': 'Multiple Extreme Values',
        'Night_High_Amount': 'Night + High Amount',
        'Anomaly_Count': 'Feature Anomaly Count'
    }
    return feature_names

if __name__ == "__main__":
    from data_collection import load_dataset
    
    # Load dataset
    df = load_dataset()
    
    # Engineer features
    df_feat, feature_cols, summary = engineer_features(df)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Total features created: {summary['total_features']}")
    print(f"   New features: {summary['new_features']}")
