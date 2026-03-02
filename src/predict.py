"""Prediction logic for invoice classification (category and tag)."""

import warnings
from datetime import datetime

import joblib

from src.config import MODEL_PATH, TAG_MODEL_PATH
from src.preprocessing import prepare_prediction_input

# Suppress benign sklearn warning about feature names in LightGBM pipeline
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Global model caches (loaded once on startup)
_category_model_cache = None
_tag_model_cache = None


def load_category_model():
    """Load category model from disk (cached globally for performance).

    Returns:
        Trained sklearn pipeline for category prediction
    """
    global _category_model_cache

    if _category_model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Category model not found at {MODEL_PATH}. "
                f"Please train the model first using: make train-category"
            )

        print(f"Loading category model from {MODEL_PATH}...")
        _category_model_cache = joblib.load(MODEL_PATH)
        print("Category model loaded successfully!")

    return _category_model_cache


def load_tag_model():
    """Load tag model from disk (cached globally for performance).

    Returns:
        Trained sklearn pipeline for tag prediction
    """
    global _tag_model_cache

    if _tag_model_cache is None:
        if not TAG_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Tag model not found at {TAG_MODEL_PATH}. "
                f"Please train the model first using: make train-tag"
            )

        print(f"Loading tag model from {TAG_MODEL_PATH}...")
        _tag_model_cache = joblib.load(TAG_MODEL_PATH)
        print("Tag model loaded successfully!")

    return _tag_model_cache


def load_model():
    """Legacy function - loads category model for backward compatibility."""
    return load_category_model()


def predict_expense_category(
    entityId: str,
    ownerId: str,
    netPrice: float,
    grossPrice: float,
    currency: str,
    invoice_title: str,
    tin: str | None,
    issueDate: str | datetime,
) -> dict[str, float]:
    """Predict expense category probabilities for an invoice.

    Args:
        entityId: Unique identifier for the company/entity
        ownerId: Unique identifier for the invoice owner
        netPrice: Net price excluding VAT
        grossPrice: Gross price including VAT
        currency: Currency code (USD, PLN, EUR, GBP, etc.)
        invoice_title: Full invoice title/description (e.g., "Adobe Systems Software Ireland Ltd")
        tin: Tax identification number (can be None)
        issueDate: Invoice issue date (YYYY-MM-DD or datetime)

    Returns:
        Dictionary mapping category names to probability scores

    Example:
        >>> predict_expense_category(
        ...     entityId="00000000-0000-0000-0000-000000000001",
        ...     ownerId="00000000-0000-0000-0000-000000000002",
        ...     netPrice=2500.0,
        ...     grossPrice=2500.0,
        ...     currency="PLN",
        ...     invoice_title="Meta Platforms Ireland Limited",
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
    pipeline = load_category_model()

    # Prepare input data
    input_data = prepare_prediction_input(
        entityId=entityId,
        ownerId=ownerId,
        netPrice=netPrice,
        grossPrice=grossPrice,
        currency=currency,
        invoice_title=invoice_title,
        tin=tin,
        issueDate=issueDate,
    )

    # Get prediction probabilities
    probabilities = pipeline.predict_proba(input_data)[0]

    # Get class labels (decode from label encoder)
    if hasattr(pipeline, "label_encoder"):
        class_labels = pipeline.label_encoder.classes_
    else:
        # Fallback: try to get from model
        class_labels = pipeline.named_steps["model"].classes_

    # Create result dictionary
    result = {
        str(label): float(prob) for label, prob in zip(class_labels, probabilities, strict=False)
    }

    # Sort by probability (highest first)
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    return result


def predict_expense_tag(
    entityId: str,
    ownerId: str,
    netPrice: float,
    grossPrice: float,
    currency: str,
    invoice_title: str,
    tin: str | None,
    issueDate: str | datetime,
) -> dict[str, float]:
    """Predict expense tag probabilities for an invoice.

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
        Dictionary mapping tag names to probability scores

    Example:
        >>> predict_expense_tag(
        ...     entityId="00000000-0000-0000-0000-000000000001",
        ...     ownerId="00000000-0000-0000-0000-000000000002",
        ...     netPrice=2500.0,
        ...     grossPrice=2500.0,
        ...     currency="PLN",
        ...     invoice_title="Meta Platforms Ireland Limited",
        ...     tin="",
        ...     issueDate="2024-08-29"
        ... )
        {
            'visual-panda': 0.85,
            'referral-fee': 0.10,
            'accounting': 0.05
        }
    """
    # Load model (cached)
    pipeline = load_tag_model()

    # Prepare input data
    input_data = prepare_prediction_input(
        entityId=entityId,
        ownerId=ownerId,
        netPrice=netPrice,
        grossPrice=grossPrice,
        currency=currency,
        invoice_title=invoice_title,
        tin=tin,
        issueDate=issueDate,
    )

    # Get prediction probabilities
    probabilities = pipeline.predict_proba(input_data)[0]

    # Get class labels (decode from label encoder)
    if hasattr(pipeline, "label_encoder"):
        class_labels = pipeline.label_encoder.classes_
    else:
        # Fallback: try to get from model
        class_labels = pipeline.named_steps["model"].classes_

    # Create result dictionary
    result = {
        str(label): float(prob) for label, prob in zip(class_labels, probabilities, strict=False)
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
    invoice_title: str,
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
        entityId, ownerId, netPrice, grossPrice, currency, invoice_title, tin, issueDate
    )

    # Get top category
    top_category = max(probabilities.items(), key=lambda x: x[1])

    return top_category


def predict_top_tag(
    entityId: str,
    ownerId: str,
    netPrice: float,
    grossPrice: float,
    currency: str,
    invoice_title: str,
    tin: str | None,
    issueDate: str | datetime,
) -> tuple[str, float]:
    """Predict the top expense tag for an invoice.

    Args:
        Same as predict_expense_tag

    Returns:
        Tuple of (tag_name, probability)

    Example:
        >>> predict_top_tag(...)
        ('visual-panda', 0.85)
    """
    probabilities = predict_expense_tag(
        entityId, ownerId, netPrice, grossPrice, currency, invoice_title, tin, issueDate
    )

    # Get top tag
    top_tag = max(probabilities.items(), key=lambda x: x[1])

    return top_tag


if __name__ == "__main__":
    # Example usage for local testing
    print("=" * 60)
    print("TESTING CATEGORY PREDICTION")
    print("=" * 60)

    category_result = predict_expense_category(
        entityId="00000000-0000-0000-0000-000000000001",
        ownerId="00000000-0000-0000-0000-000000000002",
        netPrice=2500.0,
        grossPrice=2500.0,
        currency="PLN",
        invoice_title="Meta Platforms Ireland Limited",
        tin="",
        issueDate="2024-08-29",
    )

    print("\nCategory probabilities:")
    for category, prob in category_result.items():
        print(f"  {category}: {prob:.4f}")

    top_cat, top_cat_prob = predict_top_category(
        entityId="00000000-0000-0000-0000-000000000001",
        ownerId="00000000-0000-0000-0000-000000000002",
        netPrice=2500.0,
        grossPrice=2500.0,
        currency="PLN",
        invoice_title="Meta Platforms Ireland Limited",
        tin="",
        issueDate="2024-08-29",
    )

    print(f"\nTop category: {top_cat} ({top_cat_prob:.2%})")

    print("\n" + "=" * 60)
    print("TESTING TAG PREDICTION")
    print("=" * 60)

    tag_result = predict_expense_tag(
        entityId="00000000-0000-0000-0000-000000000001",
        ownerId="00000000-0000-0000-0000-000000000002",
        netPrice=2500.0,
        grossPrice=2500.0,
        currency="PLN",
        invoice_title="Meta Platforms Ireland Limited",
        tin="",
        issueDate="2024-08-29",
    )

    print("\nTag probabilities:")
    for tag, prob in tag_result.items():
        print(f"  {tag}: {prob:.4f}")

    top_tag, top_tag_prob = predict_top_tag(
        entityId="00000000-0000-0000-0000-000000000001",
        ownerId="00000000-0000-0000-0000-000000000002",
        netPrice=2500.0,
        grossPrice=2500.0,
        currency="PLN",
        invoice_title="Meta Platforms Ireland Limited",
        tin="",
        issueDate="2024-08-29",
    )

    print(f"\nTop tag: {top_tag} ({top_tag_prob:.2%})")
