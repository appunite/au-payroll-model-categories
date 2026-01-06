"""Configuration for invoice classifier."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = MODEL_DIR / "invoice_classifier.joblib"


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables.

    Environment variables take precedence over .env file values.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Model configuration
    model_version: str = "1.0.0"

    # API configuration
    api_host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    # Optional: Database credentials (if needed for training data)
    database_url: str | None = None
    db_host: str | None = None
    db_port: int | None = None
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None


# Create global settings instance
settings = Settings()

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
    'class_weight': 'balanced',  # Handle class imbalance
}

# Export commonly used settings
MODEL_VERSION = settings.model_version
API_HOST = settings.api_host
API_PORT = settings.port
LOG_LEVEL = settings.log_level
