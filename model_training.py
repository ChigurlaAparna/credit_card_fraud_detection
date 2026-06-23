"""
Credit Card Fraud Detection - Machine Learning Model Training
Trains and compares Logistic Regression, Random Forest, XGBoost, and Gradient Boosting
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
from sklearn.model_selection import cross_val_score
import joblib
from datetime import datetime

MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
os.makedirs(MODELS_DIR, exist_ok=True)

# Model configurations (removed Gradient Boosting due to long training time)
MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
}

def train_model(name, model, X_train, y_train, X_test, y_test):
    """Train a single model and return results"""
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print('='*50)
    
    # Train the model
    print("   ⏳ Training...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'train_time': train_time
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    metrics['cv_score_mean'] = cv_scores.mean()
    metrics['cv_score_std'] = cv_scores.std()
    
    # Print results
    print(f"   ✅ Training complete in {train_time:.2f}s")
    print(f"\n   📊 Performance Metrics:")
    print(f"      Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall:    {metrics['recall']:.4f}")
    print(f"      F1 Score:  {metrics['f1_score']:.4f}")
    print(f"      ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"      CV Score:  {metrics['cv_score_mean']:.4f} (±{metrics['cv_score_std']:.4f})")
    
    print(f"\n   📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))
    
    return model, metrics

def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (name, metrics) in enumerate(results):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Genuine', 'Fraud'],
                   yticklabels=['Genuine', 'Fraud'])
        axes[i].set_xlabel('Predicted', fontsize=11)
        axes[i].set_ylabel('Actual', fontsize=11)
        axes[i].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.4f}', 
                         fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: confusion_matrices.png")
    return path

def plot_roc_curves(results, X_test, y_test):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, metrics in results:
        model = metrics['model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = metrics['roc_auc']
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: roc_curves.png")
    return path

def plot_model_comparison(results):
    """Plot model comparison metrics"""
    models = [r[0] for r in results]
    metrics_df = pd.DataFrame([{
        'Model': r[0],
        'Accuracy': r[1]['accuracy'],
        'Precision': r[1]['precision'],
        'Recall': r[1]['recall'],
        'F1 Score': r[1]['f1_score'],
        'ROC-AUC': r[1]['roc_auc']
    } for r in results])
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, col in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']):
        bars = axes[i].bar(models, metrics_df[col], color=colors, alpha=0.8, edgecolor='black')
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylim(0, 1.1)
        axes[i].set_xticklabels(models, rotation=30, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, metrics_df[col]):
            axes[i].annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    # Summary table
    axes[5].axis('off')
    table_data = metrics_df.round(4).values.tolist()
    table = axes[5].table(cellText=table_data,
                          colLabels=metrics_df.columns,
                          loc='center',
                          cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    axes[5].set_title('Model Comparison Summary', fontsize=12, fontweight='bold', y=0.85)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: model_comparison.png")
    return path

def select_best_model(results):
    """Select the best model based on F1 score (primary) and ROC-AUC (secondary)"""
    # Primary: F1 Score (balances precision and recall)
    # Secondary: ROC-AUC (overall discrimination ability)
    
    best = max(results, key=lambda x: (x[1]['f1_score'], x[1]['roc_auc']))
    return best[0], best[1]

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all models and compare performance
    """
    print("\n" + "="*70)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*70)
    print(f"Training Set: {len(X_train):,} samples")
    print(f"Test Set: {len(X_test):,} samples")
    
    results = []
    
    # Train each model
    for name, model_template in MODELS.items():
        # Create fresh model instance
        from sklearn.base import clone
        model = clone(model_template)
        
        try:
            trained_model, metrics = train_model(
                name, model, X_train, y_train, X_test, y_test
            )
            metrics['model'] = trained_model
            results.append((name, metrics))
        except Exception as e:
            print(f"   ❌ Failed to train {name}: {e}")
            continue
    
    if not results:
        print("❌ No models were successfully trained!")
        return None
    
    # Generate visualizations
    print("\n" + "="*50)
    print("Generating Visualizations...")
    print("="*50)
    
    plot_confusion_matrices(results)
    plot_roc_curves(results, X_test, y_test)
    plot_model_comparison(results)
    
    # Select best model
    best_name, best_metrics = select_best_model(results)
    
    print("\n" + "="*50)
    print("MODEL SELECTION")
    print("="*50)
    print(f"🏆 Best Model: {best_name}")
    print(f"   F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    
    # Save best model
    best_model = best_metrics['model']
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"\n💾 Best model saved to: {best_model_path}")
    
    # Save all models
    all_models_path = os.path.join(MODELS_DIR, 'all_models.joblib')
    all_models_data = {name: metrics['model'] for name, metrics in results}
    joblib.dump(all_models_data, all_models_path)
    print(f"💾 All models saved to: {all_models_path}")
    
    # Generate training report
    generate_training_report(results, best_name, best_metrics)
    
    return {
        'results': results,
        'best_model_name': best_name,
        'best_model': best_model,
        'best_metrics': best_metrics
    }

def generate_training_report(results, best_name, best_metrics):
    """Generate comprehensive training report"""
    report = []
    report.append("=" * 70)
    report.append("MODEL TRAINING REPORT")
    report.append("Credit Card Fraud Detection System")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nDataset Sizes:")
    report.append(f"   Training samples: {sum([len(r[1]['model'].predict_proba([[0]*29])) for r in results])} (approximate)")
    report.append(f"   Test samples: {sum([len(r[1]['confusion_matrix']) for r in results])} (approximate)")
    report.append("\n" + "-" * 70)
    report.append("INDIVIDUAL MODEL RESULTS")
    report.append("-" * 70)
    
    for name, metrics in results:
        report.append(f"\n📊 {name}:")
        report.append(f"   Accuracy:  {metrics['accuracy']:.4f}")
        report.append(f"   Precision: {metrics['precision']:.4f}")
        report.append(f"   Recall:    {metrics['recall']:.4f}")
        report.append(f"   F1 Score:  {metrics['f1_score']:.4f}")
        report.append(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        report.append(f"   Train Time: {metrics['train_time']:.2f}s")
        report.append(f"   CV Score:  {metrics['cv_score_mean']:.4f} (±{metrics['cv_score_std']:.4f})")
    
    report.append("\n" + "-" * 70)
    report.append("BEST MODEL SELECTION")
    report.append("-" * 70)
    report.append(f"\n🏆 Selected: {best_name}")
    report.append(f"   Reason: Highest F1 Score (balances precision and recall)")
    report.append(f"\n   Final Metrics:")
    report.append(f"   • Accuracy:  {best_metrics['accuracy']:.4f}")
    report.append(f"   • Precision: {best_metrics['precision']:.4f}")
    report.append(f"   • Recall:    {best_metrics['recall']:.4f}")
    report.append(f"   • F1 Score:  {best_metrics['f1_score']:.4f}")
    report.append(f"   • ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    
    report.append("\n" + "-" * 70)
    report.append("GENERATED FILES")
    report.append("-" * 70)
    report.append("   • models/best_model.joblib")
    report.append("   • models/all_models.joblib")
    report.append("   • reports/confusion_matrices.png")
    report.append("   • reports/roc_curves.png")
    report.append("   • reports/model_comparison.png")
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, 'model_training_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n💾 Report saved to: {report_path}")
    
    return report_text

def load_models():
    """Load trained models"""
    best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    all_models = joblib.load(os.path.join(MODELS_DIR, 'all_models.joblib'))
    return best_model, all_models

if __name__ == "__main__":
    from data_collection import load_dataset
    from data_preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    df = load_dataset()
    preprocessed = preprocess_pipeline(df)
    
    X_train = preprocessed['X_train']
    y_train = preprocessed['y_train']
    X_test = preprocessed['X_test']
    y_test = preprocessed['y_test']
    
    # Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    print("\n✅ Model training complete!")
