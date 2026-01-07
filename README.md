# Invoice Classifier

Fast ML-based invoice expense category prediction API, optimized for deployment on Google Cloud Run with minimal cold start latency.

## Features

- **High Accuracy**: 83.17% accuracy on 33 expense categories using TF-IDF text features
- **Fast Cold Starts**: Optimized for serverless deployment (~3-5 second cold starts)
- **LightGBM Model**: Faster and lighter than traditional gradient boosting
- **TF-IDF Text Processing**: Extracts semantic meaning from full invoice titles
- **REST API**: Simple JSON in/out interface via FastAPI
- **Comprehensive Metrics**: Detailed model evaluation and monitoring
- **Free Tier Friendly**: Designed to run within Google Cloud Run free tier (20-50 requests/day)
- **Environment Variables**: Support for both .env files and cloud environment variables

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

**Customizing the SQL Query** (Optional):

If your database schema differs, you can modify the query in `src/fetch_training_data.py` or provide a custom query file:

```bash
# Create custom query file
cat > my_query.sql << 'EOF'
SELECT
  entity_id as "entityId",
  owner_id as "ownerId",
  -- ... your custom query
FROM your_invoices_table
WHERE ...
EOF

# Fetch data with custom query
uv run python src/fetch_training_data.py --query-file my_query.sql
```

### Training the Model

1. **Configure database credentials**

   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env and add your PostgreSQL credentials
   # DATABASE_URL=postgresql://user:password@host:5432/database
   ```

2. **Fetch training data from PostgreSQL**

   ```bash
   # Test database connection first (optional)
   make test-db

   # Fetch training data
   make fetch-data
   ```

   This will:
   - Connect to your PostgreSQL database
   - Execute the training data query (see `src/fetch_training_data.py`)
   - Save results to `data/invoices_training_data.csv`
   - Show data statistics and category distribution

   **Note**: The query is embedded in `src/fetch_training_data.py` and can be customized if needed.

3. **Analyze and filter data (recommended)**

   ```bash
   # Analyze data distribution
   make analyze-data

   # Apply hybrid filtering (merges rare categories to parent)
   uv run python src/analyze_data.py --apply-filter hybrid
   ```

   This creates `data/invoices_training_data_filtered.csv` with better class balance.

4. **Train the model**

   ```bash
   # Option A: Train with filtered data (recommended)
   uv run python src/train_model.py invoices_training_data_filtered.csv

   # Option B: Train with original data
   make train
   ```

   This will:
   - Load and prepare the data
   - Perform 5-fold cross-validation
   - Train on full dataset
   - Save model to `models/invoice_classifier.joblib`
   - Save metrics to `models/model_metrics.json`

   **Recommended**: Use filtered data for ~1% better accuracy and 8% smaller model.

### Running Locally

```bash
# Start the API server
make run

# API will be available at http://localhost:8080
```

Test the API:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "entityId": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
    "ownerId": "e148cdec-d66d-11e9-8a40-47a686a82f23",
    "netPrice": 2500.0,
    "grossPrice": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issueDate": "2024-08-29"
  }'
```

### Testing

```bash
# Run tests
make test

# Test a single prediction
make test-predict
```

## Deployment to Google Cloud Run

### 1. Build and Test Locally

```bash
# Build Docker image
make docker-build

# Test Docker container locally
make docker-run
```

### 2. Deploy to Cloud Run

```bash
# Deploy (interactive - will prompt for service name and region)
make deploy

# Or deploy with specific settings:
gcloud run deploy payroll-invoice-classifier \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --cpu-boost \
  --timeout 60 \
  --port 8080
```

**Deployment Configuration:**
- **Memory**: 512Mi (optimal for 12MB model)
- **CPU**: 1 vCPU (sufficient for inference)
- **Min instances**: 0 (scales to zero for free tier)
- **Max instances**: 10 (handles traffic spikes)
- **CPU boost**: Enabled (reduces cold start by ~30%)
- **Timeout**: 60s (allows for cold start initialization)
- **Port**: 8080 (FastAPI default)

### 3. Integrate with Your Main Application (Recommended)

To avoid cold starts, implement keep-alive pings from your main application.

**Cold Start Performance:**
- First request after inactivity: ~9.5 seconds
- Subsequent requests (while warm): ~0.22 seconds

**Recommended Approach:** Add a background task in your main app to ping this service every 4-5 minutes while your main app is running. This ensures both services scale together automatically.

**Example (FastAPI):**
```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import httpx
import asyncio

ML_SERVICE_URL = "https://payroll-invoice-classifier-324047048236.us-central1.run.app"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background task to keep ML service warm
    task = asyncio.create_task(keep_ml_service_warm())
    yield
    # Stop pinging on shutdown
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
                pass  # Silent failures

app = FastAPI(lifespan=lifespan)
```

**Benefits:**
- ✅ $0.00/month (no Cloud Scheduler needed)
- ✅ Automatic scaling synchronization
- ✅ ML service only warm when main app is running
- ✅ Clean, maintainable code

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint for monitoring and keep-alive.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-01-06T22:00:00"
}
```

### `POST /predict`
Predict expense category for an invoice.

**Request:**
```json
{
  "entityId": "uuid",
  "ownerId": "uuid",
  "netPrice": 2500.0,
  "grossPrice": 3075.0,
  "currency": "PLN",
  "invoice_title": "Adobe Systems Software Ireland Ltd",
  "tin": "1234567890",
  "issueDate": "2024-08-29"
}
```

**Response:**
```json
{
  "probabilities": {
    "office:rent": 0.85,
    "office:utilities": 0.10,
    "others:other": 0.05
  },
  "top_category": "office:rent",
  "top_probability": 0.85,
  "model_version": "1.0.0"
}
```

### `GET /docs`
Interactive API documentation (Swagger UI).

## Model Performance

After training, check `models/model_metrics.json` for detailed metrics:

- Cross-validation accuracy
- Test set accuracy, precision, recall, F1
- Per-class performance
- Feature importance

Example metrics:
```json
{
  "training_date": "2026-01-06T23:56:41",
  "n_samples": 4723,
  "n_classes": 33,
  "cv_mean_accuracy": 0.8232,
  "test_accuracy": 0.8317,
  "test_precision": 0.8364,
  "test_f1": 0.8323,
  "features": {
    "text_features": 200,
    "total_features": "TF-IDF + numerical + categorical"
  }
}
```

## Project Structure

```
invoice-classifier/
├── src/
│   ├── config.py              # Configuration and settings
│   ├── fetch_training_data.py # Fetch data from PostgreSQL
│   ├── train_model.py         # Model training script
│   ├── predict.py             # Prediction logic
│   └── main.py                # FastAPI application
├── tests/
│   └── test_api.py            # API tests
├── examples/
│   ├── invoice_*.json         # Example requests
│   └── test_api.sh            # Test script
├── models/                    # Trained models (gitignored)
├── data/                      # Training data (gitignored)
├── Dockerfile                 # Optimized container image
├── Makefile                   # Convenient commands
├── pyproject.toml             # Python dependencies
└── .env.example               # Environment variables template
```

## Development

```bash
# Install dev dependencies
make install

# Format code
make format

# Lint code
make lint

# Run tests
make test
```

## Performance Optimization

### Cold Start Optimization
- **Multi-stage Docker build**: Reduces image size
- **Slim Python base**: Uses `python:3.11-slim`
- **Global model caching**: Model loaded once on startup
- **Single worker**: Minimal memory footprint
- **CPU boost**: Enabled for Cloud Run deployment

### Actual Performance (Measured on Google Cloud Run)

**Cloud Run Configuration:**
- **Memory**: 512Mi
- **CPU**: 1 vCPU
- **CPU Boost**: Enabled
- **Region**: us-central1

**Cold Start Performance (after 24h suspension):**
- **Total time**: 9.54 seconds
- Breakdown:
  - Container initialization: ~3-4s
  - Model loading (12MB): ~4-5s
  - First inference: ~0.2s
  - Network overhead (DNS/TCP/TLS): ~0.2s

**Warm Request Performance:**
- **Average**: 0.22 seconds (44x faster than cold start)
- Consistent across multiple requests

**Container Metrics:**
- **Image size**: ~450MB
- **Memory usage**: ~350-400MB
- **Model size**: 12.14 MB

**Performance Impact:**
- Without keep-alive: ~9.5s cold start every 15+ minutes of inactivity
- With keep-alive from main app: ~0.2s for all requests (free, automatic scaling)

## Cost Estimation

### Google Cloud Run Free Tier
- **Requests**: 2M/month (your usage: ~1,500/month = 0.075%)
- **CPU**: 180k vCPU-seconds/month (your usage: ~300s = 0.17%)
- **Memory**: 360k GiB-seconds/month (your usage: ~150s = 0.04%)

**Result**: $0.00/month for your traffic volume

## Troubleshooting

### Model not found error
```bash
# Train the model first
make train
```

### Cold starts too slow
Implement keep-alive pings from your main application. See the "Integrate with Your Main Application" section in the deployment guide for code examples.

### Out of memory
```bash
# Increase memory in deployment
gcloud run services update payroll-invoice-classifier --memory 1Gi
```

### Port conflicts locally
```bash
# Change port in .env
PORT=8000 make run
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.
