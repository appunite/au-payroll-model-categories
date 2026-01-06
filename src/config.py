"""Configuration for invoice classifier."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = MODEL_DIR / "invoice_classifier.joblib"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Feature configuration
FEATURES = [
    'entityId', 'ownerId', 'netPrice', 'currency', 'title_normalized',
    'tin', 'issueYear', 'issueMonth', 'issueDay', 'VAT_Amount', 'VAT_Rate'
]

NUMERICAL_FEATURES = ['netPrice', 'VAT_Amount', 'VAT_Rate']
CATEGORICAL_FEATURES = ['entityId', 'ownerId', 'currency', 'title_normalized', 'tin']
DATETIME_FEATURES = ['issueYear', 'issueMonth', 'issueDay']

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# LightGBM parameters (optimized for fast inference)
LGBM_PARAMS = {
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,  # Single thread for consistent cold start performance
    'verbose': -1,
}

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
