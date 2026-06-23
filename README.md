# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20|%20RF-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive end-to-end Credit Card Fraud Detection System powered by Machine Learning and Explainable AI.

## 🚀 Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- **Automatic Model Selection**: Selects the best-performing model based on F1 Score
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling)
- **Explainable AI**: SHAP-based feature importance and prediction explanations
- **Real-time Predictions**: Flask web application with instant fraud probability
- **SQLite Database**: Stores all predictions for history and analytics
- **Dashboard**: Real-time statistics and fraud trend visualization
- **Feature Engineering**: Creates meaningful fraud-related features

## 📋 Project Structure

```
Credit_Card_Fraud_Detection/
├── data/                    # Data files
│   ├── creditcard.csv       # Dataset
│   ├── processed/           # Preprocessed data
│   └── featured/            # Engineered features
├── notebooks/               # Jupyter notebooks
├── models/                  # Trained ML models
│   ├── best_model.joblib    # Best performing model
│   └── all_models.joblib    # All trained models
├── app/                     # Flask application
│   ├── __init__.py
│   ├── database.py          # SQLite database models
│   ├── routes.py            # Application routes
│   ├── static/              # Static files
│   │   ├── css/
│   │   └── js/
│   └── templates/           # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── result.html
│       ├── dashboard.html
│       ├── history.html
│       ├── about.html
│       └── error.html
├── reports/                 # Generated reports
│   ├── eda_report.txt
│   ├── model_training_report.txt
│   ├── shap_analysis_report.txt
│   └── *.png                # Visualizations
├── tests/                   # Unit tests
├── config.py               # Configuration
├── data_collection.py      # Data loading
├── eda_analysis.py         # EDA visualizations
├── data_preprocessing.py   # Preprocessing pipeline
├── feature_engineering.py  # Feature creation
├── model_training.py       # ML model training
├── explainable_ai.py       # SHAP analysis
├── main.py                 # Main entry point
└── requirements.txt        # Dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository**
   ```bash
   cd Credit_Card_Fraud_Detection
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system**
   ```bash
   python main.py
   ```

   This will:
   - Download/load the dataset
   - Run EDA and generate visualizations
   - Preprocess the data
   - Train all ML models
   - Generate SHAP explanations
   - Launch the Flask web application

## 📊 How It Works

### 1. Data Collection
- Loads the Kaggle Credit Card Fraud Detection dataset
- 284,807 transactions with 31 features
- Class distribution: ~0.17% fraud (highly imbalanced)

### 2. Exploratory Data Analysis
- Visualizes fraud vs non-fraud distribution
- Analyzes transaction amounts and patterns
- Identifies time-based fraud patterns
- Generates correlation analysis

### 3. Data Preprocessing
- Handles missing values (if any)
- Scales numerical features using StandardScaler
- Splits data into training (80%) and testing (20%) sets
- Applies SMOTE to handle class imbalance

### 4. Feature Engineering
- Creates log-transformed amount features
- Extracts time-based features (hour, night flag)
- Generates V-feature aggregations (mean, std, min, max)
- Creates interaction features between top fraud-correlated features
- Builds risk indicator features

### 5. Model Training
Trains and compares four ML models:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with regularization
- **Gradient Boosting**: Sequential boosting

### 6. Model Evaluation
Evaluates using multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Best model is automatically selected based on F1 Score.

### 7. Explainable AI (SHAP)
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Individual prediction explanations
- Shows which features increase/decrease fraud probability

## 🎯 Web Application

### Pages

1. **Home (/)** - Transaction input form
2. **Result (/result)** - Prediction with explanations
3. **Dashboard (/dashboard)** - Statistics and trends
4. **History (/history)** - All past predictions
5. **About (/about)** - System information

### API Endpoints

- `POST /api/predict` - JSON prediction endpoint
- `GET /api/stats` - Dashboard statistics
- `GET /api/history` - Prediction history

## 📈 Dashboard Features

- Total predictions count
- Fraud vs Genuine distribution
- 7-day fraud trend chart
- Recent predictions table
- Auto-refresh every 30 seconds

## 🔍 Prediction Output

For each transaction, the system provides:
- **Fraud Probability**: 0-100%
- **Risk Level**: LOW (< 30%), MEDIUM (30-70%), HIGH (> 70%)
- **Final Prediction**: Fraud or Genuine
- **Feature Explanations**: Top factors influencing the prediction

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## 📝 Example Usage

### Web Interface

1. Open browser to `http://localhost:5000`
2. Fill in transaction details (Amount, V1-V28 features)
3. Click "Analyze Transaction"
4. View fraud probability and explanations

### Python API

```python
from app.routes import predict_fraud

transaction = {
    'V1': -0.5, 'V2': 1.2, 'V3': -0.3, 'V4': 0.5,
    'V5': -0.8, 'V6': 0.2, 'V7': 0.1, 'V8': -0.2,
    'V9': 0.5, 'V10': 0.3, 'V11': -0.1, 'V12': 0.8,
    'V13': 0.2, 'V14': -0.3, 'V15': 0.1, 'V16': 0.4,
    'V17': -0.5, 'V18': 0.2, 'V19': 0.1, 'V20': -0.2,
    'V21': 0.3, 'V22': 0.1, 'V23': -0.1, 'V24': 0.2,
    'V25': 0.1, 'V26': -0.1, 'V27': 0.0, 'V28': 0.1,
    'Amount': 150.00
}

result = predict_fraud(transaction)
print(result)
```

## 🎨 Screenshots

### Prediction Form
![Prediction Form](reports/01_class_distribution.png)

### Model Comparison
![Model Comparison](reports/model_comparison.png)

### SHAP Feature Importance
![SHAP Analysis](reports/shap_summary.png)

## 📊 Generated Reports

| Report | Description |
|--------|-------------|
| `eda_report.txt` | EDA summary and findings |
| `model_training_report.txt` | Model comparison results |
| `shap_analysis_report.txt` | SHAP feature importance |
| `confusion_matrices.png` | Model confusion matrices |
| `roc_curves.png` | ROC curves comparison |
| `model_comparison.png` | Metrics bar charts |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split ratio
TEST_SIZE = 0.2

# Risk thresholds
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.7

# Flask settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
```

## 📚 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | ML preprocessing & evaluation |
| XGBoost | Gradient boosting |
| LightGBM | Alternative boosting |
| Imbalanced-learn | SMOTE for class imbalance |
| SHAP | Explainable AI |
| Flask | Web application framework |
| SQLite | Database storage |
| Bootstrap | Frontend styling |

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/download)
- Inspired by real-world fraud detection systems
- Built with open-source tools

## 👤 Author

Built with ❤️ using Python, Machine Learning, and Flask

---

## 📞 Support

For questions or issues, please open an issue on GitHub.
