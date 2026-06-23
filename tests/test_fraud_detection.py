"""
Credit Card Fraud Detection System - Unit Tests
Tests for data processing, model training, and API endpoints
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataCollection:
    """Test data collection module"""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation"""
        from data_collection import create_synthetic_dataset
        
        df = create_synthetic_dataset()
        
        # Check basic properties
        assert df is not None
        assert len(df) > 0
        assert 'Class' in df.columns
        assert 'Amount' in df.columns
        assert 'Time' in df.columns
        
        # Check that V features exist
        for i in range(1, 29):
            assert f'V{i}' in df.columns
        
        # Check class distribution
        assert df['Class'].isin([0, 1]).all()
        
    def test_explore_dataset(self):
        """Test dataset exploration function"""
        from data_collection import create_synthetic_dataset, explore_dataset
        
        # Capture output
        df = create_synthetic_dataset()
        
        # Should run without errors
        try:
            explore_dataset(df)
            result = True
        except Exception:
            result = False
        
        assert result is True

class TestDataPreprocessing:
    """Test data preprocessing module"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 1000
        data = {
            'V1': np.random.randn(n_samples),
            'V2': np.random.randn(n_samples),
            'V3': np.random.randn(n_samples),
            'Amount': np.random.exponential(scale=100, size=n_samples),
            'Class': np.random.binomial(1, 0.1, n_samples)
        }
        return pd.DataFrame(data)
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        from data_preprocessing import handle_missing_values
        
        # Add missing values
        df = sample_data.copy()
        df.loc[0, 'V1'] = np.nan
        df.loc[1, 'Amount'] = np.nan
        
        # Should handle without error
        result = handle_missing_values(df)
        assert result is not None
        assert result['V1'].isna().sum() == 0
        assert result['Amount'].isna().sum() == 0
    
    def test_split_data(self, sample_data):
        """Test train-test split"""
        from data_preprocessing import split_data
        
        X_train, X_test, y_train, y_test = split_data(
            sample_data, test_size=0.2, random_state=42
        )
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check stratification (approximate)
        original_rate = sample_data['Class'].mean()
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - original_rate) < 0.1
        assert abs(test_rate - original_rate) < 0.1

class TestFeatureEngineering:
    """Test feature engineering module"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        data = {f'V{i}': np.random.randn(n_samples) for i in range(1, 29)}
        data['Time'] = np.random.exponential(scale=10000, size=n_samples)
        data['Amount'] = np.random.exponential(scale=100, size=n_samples)
        data['Class'] = np.random.binomial(1, 0.1, n_samples)
        return pd.DataFrame(data)
    
    def test_create_transaction_features(self, sample_data):
        """Test transaction feature creation"""
        from feature_engineering import create_transaction_features
        
        df_feat = create_transaction_features(sample_data)
        
        # Check new features
        assert 'Amount_Log' in df_feat.columns
        assert 'Hour' in df_feat.columns
        assert 'Is_Night' in df_feat.columns
        assert 'V_Mean' in df_feat.columns
        
        # Original features should still exist
        assert 'Amount' in df_feat.columns
        for i in range(1, 29):
            assert f'V{i}' in df_feat.columns
    
    def test_create_risk_indicators(self, sample_data):
        """Test risk indicator creation"""
        from feature_engineering import create_risk_indicators
        
        df_risk = create_risk_indicators(sample_data)
        
        # Check risk indicators
        assert 'Is_High_Amount' in df_risk.columns
        assert 'V14_Extreme' in df_risk.columns
        assert 'Anomaly_Count' in df_risk.columns
    
    def test_engineer_features(self, sample_data):
        """Test complete feature engineering"""
        from feature_engineering import engineer_features
        
        df_feat, feature_cols, summary = engineer_features(sample_data)
        
        # Check results
        assert df_feat is not None
        assert len(feature_cols) > len(sample_data.columns)
        assert len(summary['new_features']) > 0

class TestModelTraining:
    """Test model training module"""
    
    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 500
        X = pd.DataFrame({
            f'V{i}': np.random.randn(n_samples) for i in range(1, 11)
        })
        X['Amount'] = np.random.exponential(scale=100, size=n_samples)
        y = np.random.binomial(1, 0.3, n_samples)
        return X, y
    
    def test_model_training(self, sample_train_data):
        """Test model training"""
        from model_training import MODELS, train_model
        from sklearn.model_selection import train_test_split
        
        X, y = sample_train_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        trained_model, metrics = train_model(
            'Logistic Regression', model, X_train, y_train, X_test, y_test
        )
        
        # Check results
        assert trained_model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
    def test_select_best_model(self):
        """Test model selection"""
        from model_training import select_best_model
        
        results = [
            ('Model1', {'f1_score': 0.5, 'roc_auc': 0.8}),
            ('Model2', {'f1_score': 0.7, 'roc_auc': 0.9}),
            ('Model3', {'f1_score': 0.6, 'roc_auc': 0.85})
        ]
        
        best_name, best_metrics = select_best_model(results)
        
        assert best_name == 'Model2'
        assert best_metrics['f1_score'] == 0.7

class TestDatabase:
    """Test database functionality"""
    
    def test_prediction_model(self):
        """Test Prediction model structure"""
        # Create a mock app context for testing
        with patch('flask.Flask'):
            from app.database import Prediction
            
            # Create prediction instance
            pred = Prediction(
                amount=100.0,
                fraud_probability=0.75,
                prediction='Fraud',
                risk_level='HIGH'
            )
            
            # Check attributes
            assert pred.amount == 100.0
            assert pred.fraud_probability == 0.75
            assert pred.prediction == 'Fraud'
            assert pred.risk_level == 'HIGH'

class TestRoutes:
    """Test Flask routes"""
    
    def test_get_risk_level(self):
        """Test risk level calculation"""
        with patch('app.routes.load_model', return_value=None):
            from app.routes import get_risk_level
            
            assert get_risk_level(0.8) == "HIGH"
            assert get_risk_level(0.5) == "MEDIUM"
            assert get_risk_level(0.2) == "LOW"
            assert get_risk_level(0.3) == "MEDIUM"
            assert get_risk_level(0.7) == "HIGH"
            assert get_risk_level(0.1) == "LOW"

class TestPreprocessorFunctions:
    """Test preprocessing utility functions"""
    
    def test_preprocess_transaction(self):
        """Test transaction preprocessing"""
        with patch('app.routes.load_model', return_value=None):
            with patch('app.routes.load_preprocessors', return_value=None):
                from app.routes import preprocess_transaction
                
                transaction = {
                    'V1': 1.0, 'V2': 2.0, 'Amount': 100.0
                }
                
                result = preprocess_transaction(transaction)
                assert result is not None

class TestModelPredictions:
    """Test model prediction functionality"""
    
    @pytest.fixture
    def trained_model(self):
        """Train a simple model for testing"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model, X, y
    
    def test_prediction_output_format(self, trained_model):
        """Test prediction output format"""
        model, X, y = trained_model
        
        # Make prediction
        proba = model.predict_proba(X[:1])[0]
        
        # Check output
        assert len(proba) == 2
        assert abs(sum(proba) - 1.0) < 0.001
        assert all(0 <= p <= 1 for p in proba)

# Performance tests
class TestPerformance:
    """Test performance requirements"""
    
    def test_preprocessing_speed(self):
        """Test preprocessing completes in reasonable time"""
        import time
        from data_collection import create_synthetic_dataset
        from data_preprocessing import preprocess_pipeline
        
        df = create_synthetic_dataset()
        
        start = time.time()
        preprocess_pipeline(df, test_size=0.2, random_state=42)
        elapsed = time.time() - start
        
        # Should complete within reasonable time
        assert elapsed < 60  # 60 seconds max for preprocessing

# Integration tests
class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction flow"""
        from data_collection import create_synthetic_dataset
        from data_preprocessing import preprocess_pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Create and prepare data
        df = create_synthetic_dataset()
        result = preprocess_pipeline(df, test_size=0.2, random_state=42)
        
        X_train = result['X_train']
        y_train = result['y_train']
        X_test = result['X_test']
        y_test = result['y_test']
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make prediction
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Check results
        assert len(y_pred) == len(y_test)
        assert y_proba.shape[1] == 2

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
