#!/usr/bin/env python
"""
Run Flask app on port 12000 for public access
"""
import os
import sys

# Change to project directory
os.chdir('/workspace/project/Credit_Card_Fraud_Detection')

# Add project root to path
BASE_DIR = '/workspace/project/Credit_Card_Fraud_Detection'
sys.path.insert(0, BASE_DIR)

from app import create_app

if __name__ == '__main__':
    app = create_app()
    print("=" * 50)
    print("Credit Card Fraud Detection System")
    print("=" * 50)
    print("Starting Flask server on port 12000...")
    print("=" * 50)
    app.run(host='0.0.0.0', port=12000, debug=False, threaded=True)
