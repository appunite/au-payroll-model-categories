"""Prediction logic for invoice classification."""

import pandas as pd
import joblib
from typing import Dict
from datetime import datetime

from config import MODEL_PATH


# Global model cache (loaded once on startup)
_model_cache = None


def load_model():
    """Load model from disk (cached globally for performance).

    Returns:
        Trained sklearn pipeline
    """
    global _model_cache

    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Please train the model first using: make train"
            )

        print(f"Loading model from {MODEL_PATH}...")
        _model_cache = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")

    return _model_cache


def predict_expense_category(
    entityId: str,
    ownerId: str,
    netPrice: float,
    grossPrice: float,
    currency: str,
    title_normalized: str,
    tin: str | None,
    issueDate: str | datetime,
) -> Dict[str, float]:
    """Predict expense category probabilities for an invoice.

    Args:
        entityId: Unique identifier for the company/entity
        ownerId: Unique identifier for the invoice owner
        netPrice: Net price excluding VAT
        grossPrice: Gross price including VAT
        currency: Currency code (USD, PLN, EUR, GBP, etc.)
        title_normalized: Normalized invoice title/description
        tin: Tax identification number (can be None)
        issueDate: Invoice issue date (YYYY-MM-DD or datetime)

    Returns:
        Dictionary mapping category names to probability scores

    Example:
        >>> predict_expense_category(
        ...     entityId="c2b6df6b-35e9-4120-9e7c-d20be39d7146",
        ...     ownerId="e148cdec-d66d-11e9-8a40-47a686a82f23",
        ...     netPrice=2500.0,
        ...     grossPrice=2500.0,
        ...     currency="PLN",
        ...     title_normalized="meta",
        ...     tin="",
        ...     issueDate="2024-08-29"
        ... )
        {
            'office:rent': 0.85,
            'office:utilities': 0.10,
            'others:other': 0.05
        }
    """
    # Load model (cached)
    pipeline = load_model()

    # Convert issueDate to datetime if string
    if isinstance(issueDate, str):
        issueDate = pd.to_datetime(issueDate)
    elif not isinstance(issueDate, (pd.Timestamp, datetime)):
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

    # Create input DataFrame with exact feature names
    input_data = pd.DataFrame([{
        'entityId': entityId,
        'ownerId': ownerId,
        'netPrice': netPrice,
        'currency': currency,
        'title_normalized': title_normalized,
        'tin': tin if tin else '',  # Handle None
        'issueYear': issueYear,
        'issueMonth': issueMonth,
        'issueDay': issueDay,
        'VAT_Amount': VAT_Amount,
        'VAT_Rate': VAT_Rate,
    }])

    # Get prediction probabilities
    probabilities = pipeline.predict_proba(input_data)[0]

    # Get class labels (decode from label encoder)
    if hasattr(pipeline, 'label_encoder'):
        class_labels = pipeline.label_encoder.classes_
    else:
        # Fallback: try to get from model
        class_labels = pipeline.named_steps['model'].classes_

    # Create result dictionary
    result = {
        str(label): float(prob)
        for label, prob in zip(class_labels, probabilities)
    }

    # Sort by probability (highest first)
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    return result


def predict_top_category(
    entityId: str,
    ownerId: str,
    netPrice: float,
    grossPrice: float,
    currency: str,
    title_normalized: str,
    tin: str | None,
    issueDate: str | datetime,
) -> tuple[str, float]:
    """Predict the top expense category for an invoice.

    Args:
        Same as predict_expense_category

    Returns:
        Tuple of (category_name, probability)

    Example:
        >>> predict_top_category(...)
        ('office:rent', 0.85)
    """
    probabilities = predict_expense_category(
        entityId, ownerId, netPrice, grossPrice, currency,
        title_normalized, tin, issueDate
    )

    # Get top category
    top_category = max(probabilities.items(), key=lambda x: x[1])

    return top_category


if __name__ == "__main__":
    # Example usage for local testing
    result = predict_expense_category(
        entityId="c2b6df6b-35e9-4120-9e7c-d20be39d7146",
        ownerId="e148cdec-d66d-11e9-8a40-47a686a82f23",
        netPrice=2500.0,
        grossPrice=2500.0,
        currency="PLN",
        title_normalized="meta",
        tin="",
        issueDate="2024-08-29"
    )

    print("Prediction result:")
    for category, prob in result.items():
        print(f"  {category}: {prob:.4f}")

    top_cat, top_prob = predict_top_category(
        entityId="c2b6df6b-35e9-4120-9e7c-d20be39d7146",
        ownerId="e148cdec-d66d-11e9-8a40-47a686a82f23",
        netPrice=2500.0,
        grossPrice=2500.0,
        currency="PLN",
        title_normalized="meta",
        tin="",
        issueDate="2024-08-29"
    )

    print(f"\nTop prediction: {top_cat} ({top_prob:.2%})")
