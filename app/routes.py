"""
Credit Card Fraud Detection - Flask Application Routes
Main application logic for fraud prediction and dashboard
"""
import os
import joblib
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, jsonify
from datetime import datetime
import sys

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Create blueprint
app = Blueprint('main', __name__)

# Import database
from app.database import db, Prediction, save_prediction, get_prediction_stats, get_predictions_history, get_fraud_trend

# Load models
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_model():
    """Load the best trained model"""
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_preprocessors():
    """Load preprocessing objects"""
    try:
        from data_preprocessing import load_preprocessed_data
        return load_preprocessed_data()
    except Exception as e:
        print(f"Warning: Could not load preprocessors: {e}")
        return None

# Load model at startup
model = load_model()
preprocessors = load_preprocessors()

def get_risk_level(probability):
    """Determine risk level based on fraud probability"""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"

def preprocess_transaction(transaction_data):
    """Preprocess transaction data before prediction"""
    if preprocessors is None:
        return transaction_data
    
    scaler = preprocessors.get('scaler')
    if scaler is None:
        return transaction_data
    
    # Create a copy
    df = pd.DataFrame([transaction_data])
    
    # Scale Amount feature
    if 'Amount' in df.columns and hasattr(scaler, 'transform'):
        scale_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
        available_cols = [c for c in scale_cols if c in df.columns]
        if available_cols:
            df[available_cols] = scaler.transform(df[available_cols])
    
    return df.to_dict('records')[0]

def predict_fraud(transaction_data):
    """Make fraud prediction"""
    if model is None:
        return {
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }
    
    try:
        # Preprocess
        processed_data = preprocess_transaction(transaction_data)
        
        # Create DataFrame with correct column order
        feature_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        # Ensure all features are present
        row = []
        for feat in feature_names:
            row.append(processed_data.get(feat, 0))
        
        X = pd.DataFrame([row], columns=feature_names)
        
        # Make prediction
        probabilities = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        fraud_prob = float(probabilities[1])
        
        # Determine result
        result = {
            'success': True,
            'prediction': 'Fraud' if prediction == 1 else 'Genuine',
            'fraud_probability': fraud_prob,
            'genuine_probability': float(probabilities[0]),
            'risk_level': get_risk_level(fraud_prob),
            'model_used': 'Best Model (XGBoost/Random Forest)',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get feature contributions using SHAP-like approach
        contributions = get_feature_contributions(X)
        result['top_fraud_factors'] = contributions['fraud_increasing'][:5]
        result['top_genuine_factors'] = contributions['fraud_decreasing'][:5]
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_feature_contributions(X):
    """Get feature contributions for explanation"""
    # Simple feature importance based on model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Fallback to uniform importance
        importances = np.ones(len(X.columns)) / len(X.columns)
    
    # Create contribution dictionary
    contributions = []
    for feat, imp, val in zip(X.columns, importances, X.values[0]):
        contributions.append({
            'feature': feat,
            'value': float(val),
            'importance': float(imp)
        })
    
    # Sort by importance
    contributions = sorted(contributions, key=lambda x: x['importance'], reverse=True)
    
    return {
        'fraud_increasing': [c for c in contributions if c['value'] > 0][:5],
        'fraud_decreasing': [c for c in contributions if c['value'] <= 0][:5],
        'all': contributions
    }

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page - Prediction form"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    stats = get_prediction_stats()
    trend = get_fraud_trend()
    history = get_predictions_history(20)
    
    return render_template('dashboard.html', 
                          stats=stats, 
                          trend=trend,
                          history=history)

@app.route('/history')
def history():
    """Prediction history page"""
    predictions = get_predictions_history(100)
    return render_template('history.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    """Process fraud prediction request"""
    try:
        # Get transaction data from form
        transaction_data = {
            'V1': float(request.form.get('V1', 0)),
            'V2': float(request.form.get('V2', 0)),
            'V3': float(request.form.get('V3', 0)),
            'V4': float(request.form.get('V4', 0)),
            'V5': float(request.form.get('V5', 0)),
            'V6': float(request.form.get('V6', 0)),
            'V7': float(request.form.get('V7', 0)),
            'V8': float(request.form.get('V8', 0)),
            'V9': float(request.form.get('V9', 0)),
            'V10': float(request.form.get('V10', 0)),
            'V11': float(request.form.get('V11', 0)),
            'V12': float(request.form.get('V12', 0)),
            'V13': float(request.form.get('V13', 0)),
            'V14': float(request.form.get('V14', 0)),
            'V15': float(request.form.get('V15', 0)),
            'V16': float(request.form.get('V16', 0)),
            'V17': float(request.form.get('V17', 0)),
            'V18': float(request.form.get('V18', 0)),
            'V19': float(request.form.get('V19', 0)),
            'V20': float(request.form.get('V20', 0)),
            'V21': float(request.form.get('V21', 0)),
            'V22': float(request.form.get('V22', 0)),
            'V23': float(request.form.get('V23', 0)),
            'V24': float(request.form.get('V24', 0)),
            'V25': float(request.form.get('V25', 0)),
            'V26': float(request.form.get('V26', 0)),
            'V27': float(request.form.get('V27', 0)),
            'V28': float(request.form.get('V28', 0)),
            'Amount': float(request.form.get('Amount', 0))
        }
        
        # Make prediction
        result = predict_fraud(transaction_data)
        
        # Save to database if successful
        if result.get('success'):
            save_prediction(transaction_data, result)
        
        return render_template('result.html', result=result, transaction=transaction_data)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions"""
    try:
        data = request.get_json()
        transaction_data = data.get('transaction', {})
        
        result = predict_fraud(transaction_data)
        
        # Save to database if successful
        if result.get('success'):
            save_prediction(transaction_data, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        stats = get_prediction_stats()
        trend = get_fraud_trend()
        return jsonify({
            'success': True,
            'stats': stats,
            'trend': trend
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history')
def api_history():
    """API endpoint for prediction history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = get_predictions_history(limit)
        return jsonify({
            'success': True,
            'predictions': history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model-info')
def api_model_info():
    """API endpoint for model information"""
    try:
        return jsonify({
            'success': True,
            'model_loaded': model is not None,
            'model_type': type(model).__name__ if model else None,
            'preprocessors_loaded': preprocessors is not None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Internal server error'), 500

def run_app():
    """Run the Flask application"""
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    run_app()
