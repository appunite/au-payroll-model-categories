"""Shared preprocessing utilities for invoice classification models.

This module contains common preprocessing logic used by both category
and tag prediction models to ensure consistency.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare invoice features from raw dataframe.

    Adds derived features:
    - Date components (issueYear, issueMonth, issueDay)
    - VAT features (VAT_Amount, VAT_Rate)

    Args:
        df: DataFrame with columns: issueDate, grossPrice, netPrice, invoice_title

    Returns:
        DataFrame with additional feature columns
    """
    # Ensure we're working with a copy
    df = df.copy()

    # Drop rows with missing invoice_title (critical feature)
    df.dropna(subset=['invoice_title'], inplace=True)

    # Process issueDate
    df['issueDate'] = pd.to_datetime(df['issueDate'])
    df['issueMonth'] = df['issueDate'].dt.month
    df['issueYear'] = df['issueDate'].dt.year
    df['issueDay'] = df['issueDate'].dt.day

    # Calculate VAT features
    df['VAT_Amount'] = df['grossPrice'] - df['netPrice']
    df['VAT_Rate'] = (df['grossPrice'] / df['netPrice']) - 1

    # Handle infinite values
    df.loc[df['VAT_Rate'].isin([float('inf'), float('-inf')]), 'VAT_Rate'] = 0

    return df


def create_preprocessing_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    datetime_features: list[str],
    text_feature: str,
    max_tfidf_features: int = 200
) -> ColumnTransformer:
    """Create sklearn preprocessing pipeline with TF-IDF for text.

    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        datetime_features: List of datetime-derived column names
        text_feature: Name of text column for TF-IDF
        max_tfidf_features: Maximum number of TF-IDF features

    Returns:
        ColumnTransformer for preprocessing
    """
    # Numerical transformer
    numerical_transformer = SimpleImputer(strategy='median')

    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            max_categories=50
        ))
    ])

    # Text transformer: TF-IDF on invoice titles
    # Key parameters:
    # - max_features: limit vocab size for faster inference
    # - ngram_range: capture 1-2 word phrases (e.g., "software license")
    # - min_df: ignore very rare terms (appear in <3 documents)
    # - max_df: ignore very common terms (appear in >80% of documents)
    text_transformer = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=3,  # Must appear in at least 3 documents
        max_df=0.8,  # Must not appear in >80% of documents
        lowercase=True,
        strip_accents='unicode',  # Handle Polish characters
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Words with 2+ letters
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('datetime', 'passthrough', datetime_features),
            ('text', text_transformer, text_feature),
        ]
    )

    return preprocessor


def prepare_prediction_input(
    entityId: str,
    ownerId: str,
    netPrice: float,
    grossPrice: float,
    currency: str,
    invoice_title: str,
    tin: str | None,
    issueDate: str | pd.Timestamp
) -> pd.DataFrame:
    """Prepare a single invoice record for prediction.

    Args:
        entityId: Unique identifier for the company/entity
        ownerId: Unique identifier for the invoice owner
        netPrice: Net price excluding VAT
        grossPrice: Gross price including VAT
        currency: Currency code (USD, PLN, EUR, GBP, etc.)
        invoice_title: Full invoice title/description
        tin: Tax identification number (can be None)
        issueDate: Invoice issue date (YYYY-MM-DD or datetime)

    Returns:
        DataFrame with single row ready for model prediction
    """
    # Convert issueDate to datetime if string
    if isinstance(issueDate, str):
        issueDate = pd.to_datetime(issueDate)
    elif not isinstance(issueDate, pd.Timestamp):
        issueDate = pd.to_datetime(issueDate)

    # Extract date components
    issueYear = issueDate.year
    issueMonth = issueDate.month
    issueDay = issueDate.day

    # Calculate VAT features
    VAT_Amount = grossPrice - netPrice
    VAT_Rate = (grossPrice / netPrice) - 1 if netPrice != 0 else 0

    # Handle edge cases
    if VAT_Rate in [float('inf'), float('-inf')]:
        VAT_Rate = 0

    # Create input DataFrame with exact feature names and order from training
    input_data = pd.DataFrame([{
        # Numerical features
        'netPrice': netPrice,
        'VAT_Amount': VAT_Amount,
        'VAT_Rate': VAT_Rate,
        # Categorical features
        'entityId': entityId,
        'ownerId': ownerId,
        'currency': currency,
        'tin': tin if tin else '',  # Handle None
        # Datetime features
        'issueYear': issueYear,
        'issueMonth': issueMonth,
        'issueDay': issueDay,
        # Text feature
        'invoice_title': invoice_title,
    }])

    return input_data


# Feature definitions (used by both models)
NUMERICAL_FEATURES = ['netPrice', 'VAT_Amount', 'VAT_Rate']
CATEGORICAL_FEATURES = ['entityId', 'ownerId', 'currency', 'tin']
DATETIME_FEATURES = ['issueYear', 'issueMonth', 'issueDay']
TEXT_FEATURE = 'invoice_title'

# All features in order
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DATETIME_FEATURES + [TEXT_FEATURE]
