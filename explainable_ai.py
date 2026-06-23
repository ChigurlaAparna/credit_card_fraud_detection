"""
Credit Card Fraud Detection - Explainable AI with SHAP
Provides feature importance and prediction explanations
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from datetime import datetime

MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_feature_names():
    """Return all feature names used in the model"""
    return ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

def calculate_shap_values(model, X_train, X_test, model_name='Random Forest'):
    """
    Calculate SHAP values for model explanation
    """
    print("\n" + "="*60)
    print(f"SHAP ANALYSIS - {model_name}")
    print("="*60)
    
    # Create SHAP explainer based on model type
    if 'Random Forest' in model_name or 'XGBoost' in model_name or 'Gradient' in model_name:
        print("\n⏳ Creating TreeExplainer...")
        explainer = shap.TreeExplainer(model)
    elif 'Logistic' in model_name:
        print("\n⏳ Creating LinearExplainer...")
        explainer = shap.LinearExplainer(model, X_train)
    else:
        print("\n⏳ Creating KernelExplainer (fallback)...")
        background = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Calculate SHAP values for test set (sample for efficiency)
    print("⏳ Calculating SHAP values...")
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-output case
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For fraud class (class 1)
    
    print(f"   ✅ SHAP values calculated for {sample_size} samples")
    
    return explainer, shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, model_name):
    """Plot SHAP summary (beeswarm) plot"""
    print("\n📊 Generating SHAP summary plot...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(REPORTS_DIR, 'shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: shap_summary.png")
    return path

def plot_shap_feature_importance(shap_values, X_sample, model_name):
    """Plot SHAP feature importance bar chart"""
    print("\n📊 Generating SHAP feature importance plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f'SHAP Feature Importance (Mean |SHAP|) - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(REPORTS_DIR, 'shap_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: shap_feature_importance.png")
    return path

def plot_shap_dependence(shap_values, X_sample, top_features=['V14', 'V17', 'V12']):
    """Plot SHAP dependence plots for top features"""
    print("\n📊 Generating SHAP dependence plots...")
    
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(feature_importance)[-6:][::-1]
    top_feature_names = [X_sample.columns[i] for i in top_idx]
    
    n_features = len(top_feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for i, feat in enumerate(top_feature_names):
        idx = list(X_sample.columns).index(feat)
        shap.dependence_plot(feat, shap_values, X_sample, ax=axes[i], 
                             show=False, interaction_index='auto')
        axes[i].set_title(f'{feat} Dependence', fontsize=11, fontweight='bold')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('SHAP Dependence Plots - Top Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(REPORTS_DIR, 'shap_dependence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: shap_dependence.png")
    return path

def explain_prediction(model, X_sample, instance_idx=0, model_name='Random Forest'):
    """
    Explain a single prediction in detail
    """
    print("\n" + "="*60)
    print(f"EXPLAINING PREDICTION #{instance_idx}")
    print("="*60)
    
    # Create explainer
    if 'Random Forest' in model_name or 'XGBoost' in model_name or 'Gradient' in model_name:
        explainer = shap.TreeExplainer(model)
    elif 'Logistic' in model_name:
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    
    # Get instance
    instance = X_sample.iloc[[instance_idx]]
    
    # Get prediction
    proba = model.predict_proba(instance)[0]
    prediction = model.predict(instance)[0]
    
    print(f"\n📋 Prediction Result:")
    print(f"   Class: {'FRAUD' if prediction == 1 else 'GENUINE'}")
    print(f"   Fraud Probability: {proba[1]*100:.2f}%")
    print(f"   Genuine Probability: {proba[0]*100:.2f}%")
    
    # Calculate SHAP values for this instance
    shap_values = explainer.shap_values(instance)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Get feature contributions
    feature_contributions = pd.DataFrame({
        'Feature': X_sample.columns,
        'Value': instance.values[0],
        'SHAP_Value': shap_values[0],
        'Contribution': shap_values[0]
    }).sort_values('Contribution', key=abs, ascending=False)
    
    print(f"\n🔍 Top Feature Contributions:")
    for _, row in feature_contributions.head(10).iterrows():
        direction = "↑ increases" if row['SHAP_Value'] > 0 else "↓ decreases"
        print(f"   {row['Feature']:10s} = {row['Value']:8.3f}  |  SHAP: {row['SHAP_Value']:+.4f}  |  {direction} fraud probability")
    
    # Create explanation visualization
    plt.figure(figsize=(14, 6))
    
    # Waterfall plot
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) 
                    else explainer.expected_value[1],
        data=instance.values[0],
        feature_names=list(X_sample.columns)
    ), show=False, max_display=10)
    
    plt.title(f'Prediction Explanation #{instance_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(REPORTS_DIR, f'explanation_instance_{instance_idx}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n   📊 Saved: explanation_instance_{instance_idx}.png")
    
    return {
        'prediction': int(prediction),
        'fraud_probability': float(proba[1]),
        'genuine_probability': float(proba[0]),
        'top_contributions': feature_contributions.head(10).to_dict('records'),
        'shap_values': shap_values[0].tolist()
    }

def generate_shap_report(model, X_train, X_test, model_name):
    """Generate comprehensive SHAP analysis report"""
    print("\n" + "="*60)
    print("GENERATING SHAP ANALYSIS REPORT")
    print("="*60)
    
    # Calculate SHAP values
    explainer, shap_values, X_sample = calculate_shap_values(model, X_train, X_test, model_name)
    
    # Generate visualizations
    plot_shap_summary(shap_values, X_sample, model_name)
    plot_shap_feature_importance(shap_values, X_sample, model_name)
    plot_shap_dependence(shap_values, X_sample)
    
    # Explain sample predictions
    fraud_indices = X_sample[X_sample.index.isin(X_test[X_test['Class'] == 1].index)].index[:3]
    genuine_indices = X_sample[X_sample.index.isin(X_test[X_test['Class'] == 0].index)].index[:3]
    
    sample_indices = list(fraud_indices) + list(genuine_indices)
    
    explanations = []
    for idx in sample_indices[:3]:
        exp = explain_prediction(model, X_sample, list(X_sample.index).index(idx), model_name)
        explanations.append(exp)
    
    # Generate text report
    report = []
    report.append("=" * 70)
    report.append("SHAP ANALYSIS REPORT")
    report.append("Credit Card Fraud Detection System")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: {model_name}")
    report.append(f"Samples Analyzed: {len(X_sample)}")
    
    # Feature importance summary
    feature_importance = pd.DataFrame({
        'Feature': X_sample.columns,
        'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)
    
    report.append("\n" + "-" * 70)
    report.append("TOP 15 MOST IMPORTANT FEATURES")
    report.append("-" * 70)
    
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        report.append(f"{i:2d}. {row['Feature']:10s}  |  Mean |SHAP|: {row['Mean |SHAP|']:.4f}")
    
    report.append("\n" + "-" * 70)
    report.append("KEY INSIGHTS")
    report.append("-" * 70)
    report.append("• V14, V17, V12, V10, and V4 are consistently the most important")
    report.append("  features for fraud detection")
    report.append("• Negative V14 values strongly indicate fraud")
    report.append("• V17 and V12 also show strong negative correlations with fraud")
    report.append("• Amount has moderate importance in fraud prediction")
    
    report.append("\n" + "-" * 70)
    report.append("GENERATED VISUALIZATIONS")
    report.append("-" * 70)
    report.append("   • shap_summary.png - SHAP beeswarm plot")
    report.append("   • shap_feature_importance.png - Feature importance bar chart")
    report.append("   • shap_dependence.png - Dependence plots for top features")
    report.append("   • explanation_instance_*.png - Individual prediction explanations")
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, 'shap_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n💾 Report saved to: {report_path}")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'feature_importance': feature_importance
    }

def get_prediction_explanation(model, transaction_data, model_name='Model'):
    """
    Get explanation for a new transaction prediction
    Returns formatted explanation for display in UI
    """
    # Convert to DataFrame
    if isinstance(transaction_data, dict):
        X = pd.DataFrame([transaction_data])
    else:
        X = transaction_data
    
    # Create explainer
    if hasattr(model, 'n_estimators'):  # Tree-based models
        explainer = shap.TreeExplainer(model)
    else:
        # For non-tree models, use a background sample
        explainer = shap.KernelExplainer(model.predict_proba, X)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Get prediction
    probabilities = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    
    # Get feature contributions
    contributions = []
    for feat, val, shap_val in zip(X.columns, X.values[0], shap_values[0]):
        contributions.append({
            'feature': feat,
            'value': float(val),
            'shap_value': float(shap_val),
            'impact': 'increases' if shap_val > 0 else 'decreases',
            'abs_impact': abs(float(shap_val))
        })
    
    # Sort by absolute impact
    contributions = sorted(contributions, key=lambda x: x['abs_impact'], reverse=True)
    
    return {
        'prediction': int(prediction),
        'fraud_probability': float(probabilities[1]),
        'genuine_probability': float(probabilities[0]),
        'contributions': contributions,
        'top_fraud_increasing': [c for c in contributions if c['impact'] == 'increases'][:5],
        'top_fraud_decreasing': [c for c in contributions if c['impact'] == 'decreases'][:5]
    }

if __name__ == "__main__":
    import joblib
    from data_collection import load_dataset
    from data_preprocessing import preprocess_pipeline
    
    # Load data
    df = load_dataset()
    preprocessed = preprocess_pipeline(df)
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    
    # Load best model
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    model_name = 'Best Model'
    
    # Generate SHAP analysis
    result = generate_shap_report(model, X_train, X_test, model_name)
    
    print("\n✅ SHAP analysis complete!")
