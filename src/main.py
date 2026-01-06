"""FastAPI application for invoice classification."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime
import logging

from predict import predict_expense_category, load_model
from config import MODEL_VERSION, API_HOST, API_PORT, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Invoice Classifier API",
    description="ML-based invoice expense category prediction",
    version=MODEL_VERSION,
)


# Request/Response models
class InvoiceRequest(BaseModel):
    """Invoice data for classification."""
    entityId: str = Field(..., description="Company/entity unique identifier")
    ownerId: str = Field(..., description="Invoice owner unique identifier")
    netPrice: float = Field(..., gt=0, description="Net price (excluding VAT)")
    grossPrice: float = Field(..., gt=0, description="Gross price (including VAT)")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code (e.g., PLN, USD, EUR)")
    invoice_title: str = Field(..., min_length=1, description="Full invoice title/description")
    tin: Optional[str] = Field(None, description="Tax identification number (optional)")
    issueDate: str = Field(..., description="Invoice issue date (YYYY-MM-DD)")

    class Config:
        json_schema_extra = {
            "example": {
                "entityId": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
                "ownerId": "e148cdec-d66d-11e9-8a40-47a686a82f23",
                "netPrice": 2500.0,
                "grossPrice": 3075.0,
                "currency": "PLN",
                "invoice_title": "Adobe Systems Software Ireland Ltd",
                "tin": "1234567890",
                "issueDate": "2024-08-29"
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


# Load model on startup (important for cold start optimization!)
@app.on_event("startup")
async def startup_event():
    """Load model into memory on startup."""
    logger.info("Starting Invoice Classifier API...")
    logger.info(f"Model version: {MODEL_VERSION}")

    try:
        load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
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
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring and keep-alive."""
    try:
        model = load_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(invoice: InvoiceRequest):
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
            entityId=invoice.entityId,
            ownerId=invoice.ownerId,
            netPrice=invoice.netPrice,
            grossPrice=invoice.grossPrice,
            currency=invoice.currency,
            invoice_title=invoice.invoice_title,
            tin=invoice.tin,
            issueDate=invoice.issueDate,
        )

        # Extract top prediction
        top_category = list(probabilities.keys())[0]
        top_probability = list(probabilities.values())[0]

        logger.info(f"Prediction: {top_category} ({top_probability:.2%})")

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
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=503, detail="Model not available")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL,
    )
