"""FastAPI application for invoice classification."""

import logging
import re
import secrets
import time
from collections import defaultdict
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from src.config import (
    API_HOST,
    API_PORT,
    API_TOKEN,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_REQUESTS,
    LOG_RESPONSES,
    MODEL_VERSION,
    RATE_LIMIT_RPM,
)
from src.logging_utils import (
    RequestIDMiddleware,
    TimingMiddleware,
    log_request_details,
    log_response_details,
    setup_logging,
)
from src.predict import (
    load_category_model,
    load_tag_model,
    predict_expense_category,
    predict_expense_tag,
)

# Configure logging with request ID support
setup_logging(LOG_LEVEL, LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Bearer token auth ---
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> None:
    """Validate Bearer token. Uses constant-time comparison to prevent timing attacks."""
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, API_TOKEN
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


# --- In-memory per-IP rate limiter ---
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_last_cleanup: float = time.monotonic()
_CLEANUP_INTERVAL = 60.0  # seconds


def _cleanup_rate_limit_store() -> None:
    """Remove entries older than 60 seconds."""
    global _rate_limit_last_cleanup
    now = time.monotonic()
    if now - _rate_limit_last_cleanup < _CLEANUP_INTERVAL:
        return
    _rate_limit_last_cleanup = now
    cutoff = now - 60.0
    for ip in list(_rate_limit_store):
        _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > cutoff]
        if not _rate_limit_store[ip]:
            del _rate_limit_store[ip]


async def check_rate_limit(request: Request) -> None:
    """Enforce per-IP rate limiting on prediction endpoints."""
    _cleanup_rate_limit_store()
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - 60.0
    timestamps = _rate_limit_store[client_ip]
    # Trim old entries for this IP
    _rate_limit_store[client_ip] = [t for t in timestamps if t > cutoff]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_limit_store[client_ip].append(now)


# Combined dependency list for prediction endpoints
_predict_deps = [Depends(verify_token), Depends(check_rate_limit)]

# OpenAPI tag metadata
openapi_tags = [
    {
        "name": "Prediction",
        "description": "Invoice classification endpoints. Predict expense categories (36 classes) or tags (17 classes).",
    },
    {
        "name": "Health",
        "description": "Service health monitoring and API information.",
    },
]

# Create FastAPI app
app = FastAPI(
    title="Invoice Classifier API",
    description=(
        "ML-based invoice expense classification API with dual-model architecture. "
        "Predicts expense categories (36 classes) and tags (17 classes) using LightGBM "
        "models with TF-IDF text features. Optimized for serverless deployment on Google Cloud Run."
    ),
    version=MODEL_VERSION,
    openapi_tags=openapi_tags,
)

# Add middleware (order matters - last added is executed first)
# So we add Timing first, then RequestID
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)


# Request/Response models
class InvoiceRequest(BaseModel):
    """Invoice data for classification."""

    entity_id: str = Field(..., description="Company/entity unique identifier")
    owner_id: str = Field(..., description="Invoice owner unique identifier")
    net_price: float = Field(..., gt=0, description="Net price (excluding VAT)")
    gross_price: float = Field(..., gt=0, description="Gross price (including VAT)")
    currency: str = Field(
        ..., min_length=3, max_length=3, description="Currency code (e.g., PLN, USD, EUR)"
    )
    invoice_title: str = Field(..., min_length=1, description="Full invoice title/description")
    tin: str | None = Field(None, description="Tax identification number (optional)")
    issue_date: str = Field(..., description="Invoice issue date (YYYY-MM-DD)")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code against common ISO 4217 codes."""
        # Common currency codes used in the system
        VALID_CURRENCIES = {
            "PLN",
            "USD",
            "EUR",
            "GBP",
            "CHF",
            "CZK",
            "DKK",
            "SEK",
            "NOK",
            "CAD",
            "AUD",
            "JPY",
            "CNY",
            "INR",
            "BRL",
            "MXN",
            "ZAR",
            "SGD",
            "HKD",
            "NZD",
            "KRW",
            "TRY",
            "RUB",
            "AED",
            "SAR",
            "THB",
            "MYR",
            "IDR",
            "PHP",
            "VND",
            "ILS",
            "RON",
            "HUF",
            "BGN",
            "HRK",
            "ISK",
        }

        v_upper = v.upper()
        if v_upper not in VALID_CURRENCIES:
            raise ValueError(
                f"Invalid currency code '{v}'. Must be a valid ISO 4217 code "
                f"(e.g., PLN, USD, EUR, GBP)"
            )
        return v_upper

    @field_validator("issue_date")
    @classmethod
    def validate_issue_date(cls, v: str) -> str:
        """Validate issue_date format and reasonable range."""
        # Check format (YYYY-MM-DD)
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Invalid date format '{v}'. Must be YYYY-MM-DD (e.g., 2024-08-29)")

        # Try to parse the date
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date '{v}'. {str(e)}") from e

        # Check reasonable range (2000 to present, not future)
        min_date = datetime(2000, 1, 1)
        max_date = datetime.now()

        if date_obj < min_date:
            raise ValueError(f"Date '{v}' is too old. Must be after 2000-01-01")

        if date_obj > max_date:
            raise ValueError(f"Date '{v}' is in the future. Must not be later than today")

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
                "issue_date": "2024-08-29",
            }
        }


class CategoryPredictionResponse(BaseModel):
    """Prediction result with category probabilities."""

    probabilities: dict[str, float] = Field(
        ..., description="Category probabilities (sorted by confidence)"
    )
    top_category: str = Field(..., description="Most likely category")
    top_probability: float = Field(..., description="Confidence score for top category")
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "probabilities": {
                    "office:rent": 0.85,
                    "office:utilities": 0.10,
                    "others:other": 0.05,
                },
                "top_category": "office:rent",
                "top_probability": 0.85,
                "model_version": "1.0.0",
            }
        }


class TagPredictionResponse(BaseModel):
    """Prediction result with tag probabilities."""

    probabilities: dict[str, float] = Field(
        ..., description="Tag probabilities (sorted by confidence)"
    )
    top_tag: str = Field(..., description="Most likely tag")
    top_probability: float = Field(..., description="Confidence score for top tag")
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "probabilities": {
                    "legal-advice": 0.75,
                    "benefit-training": 0.15,
                    "accounting": 0.10,
                },
                "top_tag": "legal-advice",
                "top_probability": 0.75,
                "model_version": "1.0.0",
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
@app.get(
    "/",
    response_model=dict,
    tags=["Health"],
    operation_id="get_api_info",
    summary="API information",
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Invoice Classifier API",
        "version": MODEL_VERSION,
        "endpoints": {
            "predict_category": "/predict/category",
            "predict_tag": "/predict/tag",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    operation_id="health_check",
    summary="Health check",
)
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
        timestamp=datetime.now().isoformat(),
    )


@app.post(
    "/predict/category",
    response_model=CategoryPredictionResponse,
    tags=["Prediction"],
    operation_id="predict_category",
    summary="Predict expense category",
    dependencies=_predict_deps,
    responses={
        401: {"description": "Invalid or missing API token"},
        429: {"description": "Rate limit exceeded"},
        400: {"description": "Invalid input data (e.g., negative price, invalid currency)"},
        503: {"description": "Category model not loaded"},
        500: {"description": "Internal prediction error"},
    },
)
async def predict_category(invoice: InvoiceRequest, request: Request):
    """Predict expense category for an invoice.

    Args:
        invoice: Invoice data
        request: FastAPI request object (for request_id)

    Returns:
        Prediction with category probabilities

    Raises:
        HTTPException: If prediction fails
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        logger.info(f"Predicting category for invoice: {invoice.invoice_title[:50]}...")

        # Log full request details if enabled
        if LOG_REQUESTS:
            log_request_details(logger, invoice.model_dump(), request_id)

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

        # Build response
        response_data = CategoryPredictionResponse(
            probabilities=probabilities,
            top_category=top_category,
            top_probability=top_probability,
            model_version=MODEL_VERSION,
        )

        # Log full response details if enabled
        if LOG_RESPONSES:
            log_response_details(logger, response_data.model_dump(), request_id)

        return response_data

    except ValueError as e:
        logger.error(f"Validation error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        logger.error(f"Category model not found: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Category model not available") from e
    except Exception as e:
        logger.error(
            f"Category prediction error: {e}", exc_info=True, extra={"request_id": request_id}
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


@app.post(
    "/predict/tag",
    response_model=TagPredictionResponse,
    tags=["Prediction"],
    operation_id="predict_tag",
    summary="Predict expense tag",
    dependencies=_predict_deps,
    responses={
        401: {"description": "Invalid or missing API token"},
        429: {"description": "Rate limit exceeded"},
        400: {"description": "Invalid input data (e.g., negative price, invalid currency)"},
        503: {"description": "Tag model not loaded"},
        500: {"description": "Internal prediction error"},
    },
)
async def predict_tag(invoice: InvoiceRequest, request: Request):
    """Predict expense tag for an invoice.

    Args:
        invoice: Invoice data
        request: FastAPI request object (for request_id)

    Returns:
        Prediction with tag probabilities

    Raises:
        HTTPException: If prediction fails
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        logger.info(f"Predicting tag for invoice: {invoice.invoice_title[:50]}...")

        # Log full request details if enabled
        if LOG_REQUESTS:
            log_request_details(logger, invoice.model_dump(), request_id)

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

        # Build response
        response_data = TagPredictionResponse(
            probabilities=probabilities,
            top_tag=top_tag,
            top_probability=top_probability,
            model_version=MODEL_VERSION,
        )

        # Log full response details if enabled
        if LOG_RESPONSES:
            log_response_details(logger, response_data.model_dump(), request_id)

        return response_data

    except ValueError as e:
        logger.error(f"Validation error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        logger.error(f"Tag model not found: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Tag model not available") from e
    except Exception as e:
        logger.error(f"Tag prediction error: {e}", exc_info=True, extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL,
    )
