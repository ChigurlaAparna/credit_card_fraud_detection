"""
Credit Card Fraud Detection - Flask Application Database Models
SQLite database for storing predictions and transaction history
"""
import os
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize SQLAlchemy
db = SQLAlchemy()

class Prediction(db.Model):
    """Model for storing fraud predictions"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Transaction features
    amount = db.Column(db.Float, nullable=False)
    v1 = db.Column(db.Float)
    v2 = db.Column(db.Float)
    v3 = db.Column(db.Float)
    v4 = db.Column(db.Float)
    v5 = db.Column(db.Float)
    v6 = db.Column(db.Float)
    v7 = db.Column(db.Float)
    v8 = db.Column(db.Float)
    v9 = db.Column(db.Float)
    v10 = db.Column(db.Float)
    v11 = db.Column(db.Float)
    v12 = db.Column(db.Float)
    v13 = db.Column(db.Float)
    v14 = db.Column(db.Float)
    v15 = db.Column(db.Float)
    v16 = db.Column(db.Float)
    v17 = db.Column(db.Float)
    v18 = db.Column(db.Float)
    v19 = db.Column(db.Float)
    v20 = db.Column(db.Float)
    v21 = db.Column(db.Float)
    v22 = db.Column(db.Float)
    v23 = db.Column(db.Float)
    v24 = db.Column(db.Float)
    v25 = db.Column(db.Float)
    v26 = db.Column(db.Float)
    v27 = db.Column(db.Float)
    v28 = db.Column(db.Float)
    
    # Prediction results
    fraud_probability = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    model_used = db.Column(db.String(50))
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction} ({self.fraud_probability:.2%})>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'amount': self.amount,
            'fraud_probability': self.fraud_probability,
            'prediction': self.prediction,
            'risk_level': self.risk_level,
            'model_used': self.model_used,
            'features': {
                'V1': self.v1, 'V2': self.v2, 'V3': self.v3, 'V4': self.v4,
                'V5': self.v5, 'V6': self.v6, 'V7': self.v7, 'V8': self.v8,
                'V9': self.v9, 'V10': self.v10, 'V11': self.v11, 'V12': self.v12,
                'V13': self.v13, 'V14': self.v14, 'V15': self.v15, 'V16': self.v16,
                'V17': self.v17, 'V18': self.v18, 'V19': self.v19, 'V20': self.v20,
                'V21': self.v21, 'V22': self.v22, 'V23': self.v23, 'V24': self.v24,
                'V25': self.v25, 'V26': self.v26, 'V27': self.v27, 'V28': self.v28
            }
        }

class User(db.Model):
    """Model for user authentication (optional)"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

def init_db(app):
    """Initialize database with Flask app"""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud_detection.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        print("✅ Database tables created!")

def get_prediction_stats():
    """Get prediction statistics for dashboard"""
    total = Prediction.query.count()
    fraud_count = Prediction.query.filter_by(prediction='Fraud').count()
    genuine_count = Prediction.query.filter_by(prediction='Genuine').count()
    
    # Get recent fraud probability average
    recent_predictions = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).limit(100).all()
    
    avg_fraud_prob = 0
    if recent_predictions:
        avg_fraud_prob = sum(p.fraud_probability for p in recent_predictions) / len(recent_predictions)
    
    # Get fraud trend (last 7 days)
    from datetime import timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_fraud = Prediction.query.filter(
        Prediction.timestamp >= week_ago,
        Prediction.prediction == 'Fraud'
    ).count()
    
    return {
        'total_predictions': total,
        'fraud_count': fraud_count,
        'genuine_count': genuine_count,
        'avg_fraud_probability': avg_fraud_prob,
        'recent_fraud_count': recent_fraud,
        'fraud_rate': fraud_count / total if total > 0 else 0
    }

def get_predictions_history(limit=50):
    """Get prediction history"""
    predictions = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).limit(limit).all()
    return [p.to_dict() for p in predictions]

def get_fraud_trend():
    """Get fraud trend data for charts"""
    from datetime import timedelta
    
    trend_data = []
    for i in range(7):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_total = Prediction.query.filter(
            Prediction.timestamp >= day_start,
            Prediction.timestamp < day_end
        ).count()
        
        day_fraud = Prediction.query.filter(
            Prediction.timestamp >= day_start,
            Prediction.timestamp < day_end,
            Prediction.prediction == 'Fraud'
        ).count()
        
        trend_data.append({
            'date': day_start.strftime('%Y-%m-%d'),
            'day': day_start.strftime('%A'),
            'total': day_total,
            'fraud': day_fraud,
            'fraud_rate': day_fraud / day_total if day_total > 0 else 0
        })
    
    return list(reversed(trend_data))

def save_prediction(transaction_data, prediction_result):
    """Save a prediction to the database"""
    prediction = Prediction(
        amount=transaction_data.get('Amount', 0),
        v1=transaction_data.get('V1'), v2=transaction_data.get('V2'),
        v3=transaction_data.get('V3'), v4=transaction_data.get('V4'),
        v5=transaction_data.get('V5'), v6=transaction_data.get('V6'),
        v7=transaction_data.get('V7'), v8=transaction_data.get('V8'),
        v9=transaction_data.get('V9'), v10=transaction_data.get('V10'),
        v11=transaction_data.get('V11'), v12=transaction_data.get('V12'),
        v13=transaction_data.get('V13'), v14=transaction_data.get('V14'),
        v15=transaction_data.get('V15'), v16=transaction_data.get('V16'),
        v17=transaction_data.get('V17'), v18=transaction_data.get('V18'),
        v19=transaction_data.get('V19'), v20=transaction_data.get('V20'),
        v21=transaction_data.get('V21'), v22=transaction_data.get('V22'),
        v23=transaction_data.get('V23'), v24=transaction_data.get('V24'),
        v25=transaction_data.get('V25'), v26=transaction_data.get('V26'),
        v27=transaction_data.get('V27'), v28=transaction_data.get('V28'),
        fraud_probability=prediction_result['fraud_probability'],
        prediction=prediction_result['prediction'],
        risk_level=prediction_result['risk_level'],
        model_used=prediction_result.get('model_used', 'Best Model')
    )
    
    db.session.add(prediction)
    db.session.commit()
    
    return prediction
