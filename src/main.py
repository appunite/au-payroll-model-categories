"""FastAPI application for invoice classification."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional
from datetime import datetime
import logging
import re

from src.predict import (
    predict_expense_category,
    predict_expense_tag,
    load_category_model,
    load_tag_model
)
from src.config import MODEL_VERSION, API_HOST, API_PORT, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Invoice Classifier API",
    description="ML-based invoice expense category and tag prediction",
    version=MODEL_VERSION,
)


# Request/Response models
class InvoiceRequest(BaseModel):
    """Invoice data for classification."""
    entity_id: str = Field(..., description="Company/entity unique identifier")
    owner_id: str = Field(..., description="Invoice owner unique identifier")
    net_price: float = Field(..., gt=0, description="Net price (excluding VAT)")
    gross_price: float = Field(..., gt=0, description="Gross price (including VAT)")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code (e.g., PLN, USD, EUR)")
    invoice_title: str = Field(..., min_length=1, description="Full invoice title/description")
    tin: Optional[str] = Field(None, description="Tax identification number (optional)")
    issue_date: str = Field(..., description="Invoice issue date (YYYY-MM-DD)")

    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code against common ISO 4217 codes."""
        # Common currency codes used in the system
        VALID_CURRENCIES = {
            'PLN', 'USD', 'EUR', 'GBP', 'CHF', 'CZK', 'DKK', 'SEK', 'NOK',
            'CAD', 'AUD', 'JPY', 'CNY', 'INR', 'BRL', 'MXN', 'ZAR', 'SGD',
            'HKD', 'NZD', 'KRW', 'TRY', 'RUB', 'AED', 'SAR', 'THB', 'MYR',
            'IDR', 'PHP', 'VND', 'ILS', 'RON', 'HUF', 'BGN', 'HRK', 'ISK'
        }

        v_upper = v.upper()
        if v_upper not in VALID_CURRENCIES:
            raise ValueError(
                f"Invalid currency code '{v}'. Must be a valid ISO 4217 code "
                f"(e.g., PLN, USD, EUR, GBP)"
            )
        return v_upper

    @field_validator('issue_date')
    @classmethod
    def validate_issue_date(cls, v: str) -> str:
        """Validate issue_date format and reasonable range."""
        # Check format (YYYY-MM-DD)
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(
                f"Invalid date format '{v}'. Must be YYYY-MM-DD (e.g., 2024-08-29)"
            )

        # Try to parse the date
        try:
            date_obj = datetime.strptime(v, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(
                f"Invalid date '{v}'. {str(e)}"
            )

        # Check reasonable range (2000 to present, not future)
        min_date = datetime(2000, 1, 1)
        max_date = datetime.now()

        if date_obj < min_date:
            raise ValueError(
                f"Date '{v}' is too old. Must be after 2000-01-01"
            )

        if date_obj > max_date:
            raise ValueError(
                f"Date '{v}' is in the future. Must not be later than today"
            )

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "entity_id": "00000000-0000-0000-0000-000000000001",
                "owner_id": "00000000-0000-0000-0000-000000000002",
                "net_price": 2500.0,
                "gross_price": 3075.0,
                "currency": "PLN",
                "invoice_title": "Adobe Systems Software Ireland Ltd",
                "tin": "1234567890",
                "issue_date": "2024-08-29"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result with category probabilities."""
    probabilities: Dict[str, float] = Field(..., description="Category probabilities (sorted by confidence)")
    top_category: str = Field(..., description="Most likely category")
    top_probability: float = Field(..., description="Confidence score for top category")
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "probabilities": {
                    "office:rent": 0.85,
                    "office:utilities": 0.10,
                    "others:other": 0.05
                },
                "top_category": "office:rent",
                "top_probability": 0.85,
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# Load models on startup (important for cold start optimization!)
@app.on_event("startup")
async def startup_event():
    """Load models into memory on startup."""
    logger.info("Starting Invoice Classifier API...")
    logger.info(f"Model version: {MODEL_VERSION}")

    # Load category model
    try:
        load_category_model()
        logger.info("Category model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load category model: {e}")
        # Continue startup even if model fails - health check will reflect this
        pass

    # Load tag model
    try:
        load_tag_model()
        logger.info("Tag model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load tag model: {e}")
        # Continue startup even if model fails - health check will reflect this
        pass


# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Invoice Classifier API",
        "version": MODEL_VERSION,
        "endpoints": {
            "predict_category": "/predict/category",
            "predict_tag": "/predict/tag",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring and keep-alive."""
    category_loaded = False
    tag_loaded = False

    try:
        model = load_category_model()
        category_loaded = model is not None
    except Exception:
        pass

    try:
        model = load_tag_model()
        tag_loaded = model is not None
    except Exception:
        pass

    # Both models must be loaded for healthy status
    all_loaded = category_loaded and tag_loaded

    return HealthResponse(
        status="healthy" if all_loaded else "unhealthy",
        model_loaded=all_loaded,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/category", response_model=PredictionResponse)
async def predict_category(invoice: InvoiceRequest):
    """Predict expense category for an invoice.

    Args:
        invoice: Invoice data

    Returns:
        Prediction with category probabilities

    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Predicting category for invoice: {invoice.invoice_title[:50]}...")

        # Get predictions
        probabilities = predict_expense_category(
            entityId=invoice.entity_id,
            ownerId=invoice.owner_id,
            netPrice=invoice.net_price,
            grossPrice=invoice.gross_price,
            currency=invoice.currency,
            invoice_title=invoice.invoice_title,
            tin=invoice.tin,
            issueDate=invoice.issue_date,
        )

        # Extract top prediction
        top_category = list(probabilities.keys())[0]
        top_probability = list(probabilities.values())[0]

        logger.info(f"Category prediction: {top_category} ({top_probability:.2%})")

        return PredictionResponse(
            probabilities=probabilities,
            top_category=top_category,
            top_probability=top_probability,
            model_version=MODEL_VERSION,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Category model not found: {e}")
        raise HTTPException(status_code=503, detail="Category model not available")
    except Exception as e:
        logger.error(f"Category prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/tag", response_model=PredictionResponse)
async def predict_tag(invoice: InvoiceRequest):
    """Predict expense tag for an invoice.

    Args:
        invoice: Invoice data

    Returns:
        Prediction with tag probabilities

    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Predicting tag for invoice: {invoice.invoice_title[:50]}...")

        # Get predictions
        probabilities = predict_expense_tag(
            entityId=invoice.entity_id,
            ownerId=invoice.owner_id,
            netPrice=invoice.net_price,
            grossPrice=invoice.gross_price,
            currency=invoice.currency,
            invoice_title=invoice.invoice_title,
            tin=invoice.tin,
            issueDate=invoice.issue_date,
        )

        # Extract top prediction
        top_tag = list(probabilities.keys())[0]
        top_probability = list(probabilities.values())[0]

        logger.info(f"Tag prediction: {top_tag} ({top_probability:.2%})")

        return PredictionResponse(
            probabilities=probabilities,
            top_category=top_tag,  # Reusing field name for consistency
            top_probability=top_probability,
            model_version=MODEL_VERSION,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Tag model not found: {e}")
        raise HTTPException(status_code=503, detail="Tag model not available")
    except Exception as e:
        logger.error(f"Tag prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL,
    )
