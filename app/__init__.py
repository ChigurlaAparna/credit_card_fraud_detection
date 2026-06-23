"""
Credit Card Fraud Detection - Flask Application Package
"""
from flask import Flask
from flask_cors import CORS
from app.database import db

def create_app():
    """Application factory"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'fraud-detection-secret-key-2024'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud_detection.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Register blueprints
    from app.routes import app as routes_bp
    app.register_blueprint(routes_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app
