# Invoice Classifier

Fast ML-based invoice expense category and tag prediction API, optimized for deployment on Google Cloud Run with minimal cold start latency.

## Features

- **Dual-Model Architecture**: Separate models for expense categories (36 classes) and tags (17 classes)
- **High Accuracy**: ~83% accuracy on categories using TF-IDF text features + LightGBM
- **Fast Cold Starts**: Optimized for serverless deployment (~9.5s end-to-end, ~0.2s when warm)
- **Shared Preprocessing**: Common feature engineering across both models
- **REST API**: Simple JSON in/out interface via FastAPI with snake_case field names
- **Comprehensive Logging**: Request tracking, performance metrics, structured logging (text/JSON)
- **Free Tier Friendly**: Designed to run within Google Cloud Run free tier (20-50 requests/day)

## Architecture

```text
Training Pipeline:
  Category: SQL DB → CSV → train_model_category.py → LightGBM → invoice_classifier.joblib
  Tag:      SQL DB → CSV → train_model_tag.py      → LightGBM → invoice_tag_classifier.joblib

Inference Pipeline:
  HTTP Request → FastAPI → predict.py → Model (cached) → JSON Response
                              ├─ /predict/category → category model
                              └─ /predict/tag      → tag model
```

**Key design decisions:**
- **Separate models** for category and tag — each target has different distributions and class counts
- **Shared preprocessing** (`preprocessing.py`) — both models use the same feature engineering pipeline (TF-IDF + numerical + categorical + datetime features)
- **Dual model caching** — both models loaded on startup into global caches for fast inference

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL database with invoice data
- Docker (for containerization)
- Google Cloud CLI (for deployment)
- **macOS only**: OpenMP library for LightGBM
  ```bash
  brew install libomp
  ```

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd invoice-classifier

# Install dependencies
make install

# Activate virtual environment
source .venv/bin/activate
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your PostgreSQL database credentials
# Required fields:
# - DATABASE_URL (or DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
```

### Training the Models

1. **Configure database credentials**

   ```bash
   cp .env.example .env
   # Edit .env and add your PostgreSQL credentials
   # DATABASE_URL=postgresql://user:password@host:5432/database
   ```

2. **Fetch training data from PostgreSQL**

   ```bash
   # Test database connection first (optional)
   make test-db

   # Fetch category training data
   make fetch-data
   ```

   For tag training data, run the SQL query in `queries/fetch_tag_training_data.sql` and export to `data/invoices_tag_training_data.csv`.

3. **Analyze and filter data (recommended)**

   ```bash
   make analyze-data
   uv run python src/analyze_data.py --apply-filter hybrid
   ```

4. **Train both models**

   ```bash
   # Train both category and tag models
   make train

   # Or train individually:
   make train-category
   make train-tag
   ```

   Output files:
   - `models/invoice_classifier.joblib` — category model
   - `models/invoice_tag_classifier.joblib` — tag model
   - `models/category_model_metrics.json` — category evaluation metrics
   - `models/tag_model_metrics.json` — tag evaluation metrics

### Running Locally

```bash
# Start the API server (requires both models to be trained)
make run

# API will be available at http://localhost:8080
```

Test the API:

```bash
# Category prediction (include -H "Authorization: Bearer $TOKEN" when API_TOKEN is set)
curl -X POST http://localhost:8080/predict/category \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{
    "entity_id": "00000000-0000-0000-0000-000000000001",
    "owner_id": "00000000-0000-0000-0000-000000000002",
    "net_price": 2500.0,
    "gross_price": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issue_date": "2024-08-29"
  }'

# Tag prediction
curl -X POST http://localhost:8080/predict/tag \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{
    "entity_id": "00000000-0000-0000-0000-000000000001",
    "owner_id": "00000000-0000-0000-0000-000000000002",
    "net_price": 2500.0,
    "gross_price": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issue_date": "2024-08-29"
  }'
```

> **Note:** `API_TOKEN` is required. Set it in `.env` or as an environment variable before starting the server.

### Testing

```bash
# Run tests
make test

# Test predictions locally (both models)
make test-predict
```

## Deployment to Google Cloud Run

### 1. Build and Test Locally

```bash
make docker-build
make docker-run
```

### 2. Deploy to Cloud Run

```bash
# Deploy (interactive - will prompt for service name and region)
make deploy

# Or deploy with specific settings:
gcloud run deploy payroll-invoice-classifier \
  --source . \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --cpu-boost \
  --timeout 60 \
  --port 8080
```

**Deployment Configuration:**
- **Memory**: 1Gi (two models loaded simultaneously)
- **CPU**: 1 vCPU (sufficient for inference)
- **Min instances**: 0 (scales to zero for free tier)
- **Max instances**: 10 (handles traffic spikes)
- **CPU boost**: Enabled (reduces cold start by ~30%)

### 3. Integrate with Your Main Application (Recommended)

To avoid cold starts, implement keep-alive pings from your main application:

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import httpx
import asyncio

ML_SERVICE_URL = "https://your-service.run.app"

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(keep_ml_service_warm())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

async def keep_ml_service_warm():
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            await asyncio.sleep(240)  # 4 minutes
            try:
                await client.get(f"{ML_SERVICE_URL}/health")
            except Exception:
                pass

app = FastAPI(lifespan=lifespan)
```

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint for monitoring and keep-alive. Returns `"healthy"` only when both models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-01-06T22:00:00"
}
```

### `POST /predict/category`
Predict expense category for an invoice (36 categories).

**Request:**
```json
{
  "entity_id": "00000000-0000-0000-0000-000000000001",
  "owner_id": "00000000-0000-0000-0000-000000000002",
  "net_price": 2500.0,
  "gross_price": 3075.0,
  "currency": "PLN",
  "invoice_title": "Adobe Systems Software Ireland Ltd",
  "tin": "1234567890",
  "issue_date": "2024-08-29"
}
```

**Response:**
```json
{
  "probabilities": {
    "operations:design": 0.37,
    "people:training": 0.11,
    "marketing:services": 0.10
  },
  "top_category": "operations:design",
  "top_probability": 0.37,
  "model_version": "1.0.0"
}
```

### `POST /predict/tag`
Predict expense tag for an invoice (17 tags).

**Request:** Same format as `/predict/category`.

**Response:**
```json
{
  "probabilities": {
    "legal-advice": 0.46,
    "benefit-training": 0.39,
    "esop": 0.03
  },
  "top_tag": "legal-advice",
  "top_probability": 0.46,
  "model_version": "1.0.0"
}
```

### `GET /docs`
Interactive API documentation (Swagger UI).

## Authentication & Rate Limiting

### Bearer Token Authentication

Prediction endpoints (`/predict/category`, `/predict/tag`) require a Bearer token when the `API_TOKEN` environment variable is set.

```bash
# Set the token in .env or as an environment variable
API_TOKEN=your-secret-token

# Include the token in requests
curl -X POST http://localhost:8080/predict/category \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d @examples/invoice_software.json
```

Public endpoints (`/`, `/health`, `/docs`, `/openapi.json`) do not require authentication.

`API_TOKEN` is required — the application will refuse to start without it.

### Rate Limiting

Prediction endpoints are rate-limited per IP address. Default: **60 requests/minute**.

```bash
# Configure via environment variable
RATE_LIMIT_RPM=100  # Allow 100 requests per minute per IP
```

The rate limiter is in-memory (resets on container restart), which is appropriate for Cloud Run's single-worker-per-instance architecture.

## Logging and Monitoring

The API includes comprehensive logging to help track service health and debug issues.

### Logging Features

1. **Request ID Tracking**: Every request gets a unique ID for correlation across logs
   - Returned in `X-Request-ID` response header
   - Can be provided by client via `X-Request-ID` request header

2. **Performance Metrics**: Request latency tracking
   - Returned in `X-Process-Time` response header

3. **Request/Response Logging**: Optional detailed input/output logging
   - Configurable via environment variables

4. **Structured Logging**: Support for both text and JSON formats
   - Text format: Human-readable for development
   - JSON format: Machine-parseable for production (e.g., Cloud Logging)

### Configuration

```bash
# Logging level (debug, info, warning, error)
LOG_LEVEL=info

# Enable/disable detailed logging
LOG_REQUESTS=true
LOG_RESPONSES=true
LOG_PERFORMANCE=true

# Logging format (text or json)
LOG_FORMAT=text
```

### Monitoring in Google Cloud

When deployed to Cloud Run, all logs are automatically sent to Cloud Logging where you can:
- Filter by request ID to see all logs for a specific request
- Set up alerts for error rates or latency thresholds
- Create dashboards to visualize request volume and performance

**Useful Cloud Logging Filters:**
```
severity >= ERROR
jsonPayload.duration_ms > 1000
jsonPayload.request_id = "abc-123-def"
jsonPayload.message =~ "prediction:"
```

## Model Performance

After training, check metrics files:
- `models/category_model_metrics.json` — category model evaluation
- `models/tag_model_metrics.json` — tag model evaluation

Metrics include cross-validation accuracy, test accuracy, precision, recall, F1, per-class performance, and feature importance.

## Project Structure

```text
invoice-classifier/
├── src/
│   ├── config.py                # Configuration and settings
│   ├── preprocessing.py         # Shared feature engineering
│   ├── fetch_training_data.py   # Fetch data from PostgreSQL
│   ├── analyze_data.py          # Data distribution analysis
│   ├── train_model_category.py  # Category model training
│   ├── train_model_tag.py       # Tag model training
│   ├── predict.py               # Prediction logic (both models)
│   ├── logging_utils.py         # Logging and middleware
│   └── main.py                  # FastAPI application
├── tests/
│   └── test_api.py              # API tests
├── examples/
│   ├── invoice_*.json           # Example requests
│   ├── api_responses.md         # Full API response documentation
│   └── test_api.sh              # Test script
├── queries/
│   └── fetch_tag_training_data.sql  # Tag training data query
├── models/                      # Trained models (gitignored)
├── data/                        # Training data (gitignored)
├── Dockerfile                   # Optimized container image
├── Makefile                     # Convenient commands
├── pyproject.toml               # Python dependencies
└── .env.example                 # Environment variables template
```

## Development

```bash
make install       # Install dev dependencies
make format        # Format code with ruff
make lint          # Lint code with ruff
make test          # Run tests
```

## Performance

### Cold Start (after 15+ minutes of inactivity)
- **Total time**: ~9.5 seconds
- Container initialization: ~3-4s, model loading (both models): ~4-5s, first inference: ~0.2s

### Warm Requests (with keep-alive)
- **Average**: ~0.2 seconds (44x faster than cold start)

### Container Metrics
- **Image size**: ~450MB
- **Memory usage**: ~400-500MB (two models)

## Cost Estimation

### Google Cloud Run Free Tier
- **Requests**: 2M/month (usage: ~1,500/month = 0.075%)
- **CPU**: 180k vCPU-seconds/month (usage: ~300s = 0.17%)
- **Memory**: 360k GiB-seconds/month (usage: ~150s = 0.04%)

**Result**: $0.00/month for current traffic volume.

## Troubleshooting

### Model not found error
```bash
# Train both models
make train
```

### Cold starts too slow
Implement keep-alive pings from your main application. See the deployment section above.

### Out of memory
```bash
# Increase memory (two models need more headroom)
gcloud run services update payroll-invoice-classifier --memory 1Gi
```

### Port conflicts locally
```bash
PORT=8000 make run
```

## License

MIT License - see LICENSE file for details.
