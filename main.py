"""
Credit Card Fraud Detection System - Main Entry Point
This script orchestrates the entire ML pipeline and launches the Flask app
"""
import os
import sys

def main():
    """Main entry point"""
    print("="*70)
    print("  CREDIT CARD FRAUD DETECTION SYSTEM")
    print("  Powered by Machine Learning & Explainable AI")
    print("="*70)
    
    # Check if data exists, if not run the pipeline
    models_dir = 'models'
    data_dir = 'data'
    data_file = os.path.join(data_dir, 'creditcard.csv')
    
    if not os.path.exists(os.path.join(models_dir, 'best_model.joblib')):
        print("\n⚠️ Model not found. Running full pipeline...")
        run_full_pipeline()
    
    # Launch Flask app
    print("\n🚀 Starting Flask Application...")
    from app.routes import app
    app.run(host='0.0.0.0', port=5000, debug=True)

def run_full_pipeline():
    """Run the complete ML pipeline"""
    print("\n" + "="*70)
    print("  RUNNING FULL ML PIPELINE")
    print("="*70)
    
    # Step 1: Data Collection
    print("\n📥 Step 1: Data Collection")
    print("-"*40)
    from data_collection import download_dataset, explore_dataset
    df = download_dataset()
    explore_dataset(df)
    
    # Step 2: EDA
    print("\n📊 Step 2: Exploratory Data Analysis")
    print("-"*40)
    from eda_analysis import generate_eda_report
    generate_eda_report(df)
    
    # Step 3: Preprocessing
    print("\n⚙️ Step 3: Data Preprocessing")
    print("-"*40)
    from data_preprocessing import preprocess_pipeline
    preprocessed = preprocess_pipeline(df)
    
    # Step 4: Feature Engineering
    print("\n🔧 Step 4: Feature Engineering")
    print("-"*40)
    from feature_engineering import engineer_features
    df_feat, feature_cols, summary = engineer_features(df)
    
    # Step 5: Model Training
    print("\n🤖 Step 5: Model Training")
    print("-"*40)
    from model_training import train_all_models
    X_train = preprocessed['X_train']
    y_train = preprocessed['y_train']
    X_test = preprocessed['X_test']
    y_test = preprocessed['y_test']
    
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 6: SHAP Analysis
    print("\n🧠 Step 6: Explainable AI (SHAP)")
    print("-"*40)
    from explainable_ai import generate_shap_report
    model = results['best_model']
    generate_shap_report(model, X_train, X_test, results['best_model_name'])
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n✅ Best Model: {results['best_model_name']}")
    print(f"   F1 Score: {results['best_metrics']['f1_score']:.4f}")
    print(f"   ROC-AUC: {results['best_metrics']['roc_auc']:.4f}")
    print(f"\n📁 Models saved to: {models_dir}/")
    print(f"📁 Reports saved to: reports/")
    print("="*70)

if __name__ == '__main__':
    main()
