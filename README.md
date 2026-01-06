# Invoice Classifier

Fast ML-based invoice expense category prediction API, optimized for deployment on Google Cloud Run with minimal cold start latency.

## Features

- **Fast Cold Starts**: Optimized for serverless deployment (~3-5 second cold starts)
- **LightGBM Model**: Faster and lighter than traditional gradient boosting
- **REST API**: Simple JSON in/out interface via FastAPI
- **Comprehensive Metrics**: Detailed model evaluation and monitoring
- **Free Tier Friendly**: Designed to run within Google Cloud Run free tier (20-50 requests/day)
- **Environment Variables**: Support for both .env files and cloud environment variables

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for containerization)
- Google Cloud CLI (for deployment)

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

# Edit .env with your configuration (optional)
# Most defaults work out of the box
```

### Training the Model

1. **Export training data from your database**

   Run your SQL query and export results to `data/invoices_training_data.csv`:

   ```sql
   SELECT
       i."entityId",
       i."ownerId",
       i."issueDate",
       i."netPrice",
       i."grossPrice",
       i.currency,
       i."titleNormalized" as "title_normalized",
       regexp_replace(
           regexp_replace(
               CASE
                   WHEN bt."beneficiaryTin" IS NOT NULL THEN bt."beneficiaryTin"
                   WHEN length(t."documentData") > 0 THEN json_entity.value->>'mentionText'
                   ELSE NULL
               END,
               '^\s*PL', '', 'gi'
           ),
           '[\s\-–—]', '', 'g'
       ) AS tin,
       i."expenseCategory"
   FROM invoices i
   -- ... (rest of your query)
   ```

2. **Train the model**

   ```bash
   make train
   ```

   This will:
   - Load and prepare the data
   - Perform 5-fold cross-validation
   - Train on full dataset
   - Save model to `models/invoice_classifier.joblib`
   - Save metrics to `models/model_metrics.json`

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
    "title_normalized": "office rent january",
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
```

Default deployment configuration:
- **Memory**: 512Mi
- **CPU**: 1
- **Min instances**: 0 (scales to zero)
- **Max instances**: 10
- **CPU boost**: Enabled (for faster cold starts)
- **Timeout**: 60s

### 3. Set Up Keep-Warm Scheduler (Optional but Recommended)

To minimize cold starts for your low-traffic use case (~20-50 requests/day):

```bash
# After deployment, set up Cloud Scheduler
make setup-scheduler
```

This creates a Cloud Scheduler job that pings your service every 5 minutes to keep it warm.

**Cost**: ~$0.10/month (essentially free)
**Benefit**: Most requests will be fast (50-200ms instead of 3-5 seconds)

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
  "title_normalized": "office rent",
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
  "training_date": "2026-01-06T22:00:00",
  "n_samples": 5000,
  "cv_mean_accuracy": 0.92,
  "test_accuracy": 0.91,
  "test_precision": 0.90,
  "test_f1": 0.91
}
```

## Project Structure

```
invoice-classifier/
├── src/
│   ├── config.py          # Configuration and settings
│   ├── train_model.py     # Model training script
│   ├── predict.py         # Prediction logic
│   └── main.py            # FastAPI application
├── tests/
│   └── test_api.py        # API tests
├── models/                # Trained models (gitignored)
├── data/                  # Training data (gitignored)
├── Dockerfile             # Optimized container image
├── Makefile               # Convenient commands
├── pyproject.toml         # Python dependencies
└── .env.example           # Environment variables template
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

### Expected Performance
- **Cold start**: 3-5 seconds
- **Warm request**: 50-200ms
- **Image size**: ~300-500MB
- **Memory usage**: ~300-400MB

## Cost Estimation

### Google Cloud Run Free Tier
- **Requests**: 2M/month (your usage: ~1,500/month = 0.075%)
- **CPU**: 180k vCPU-seconds/month (your usage: ~300s = 0.17%)
- **Memory**: 360k GiB-seconds/month (your usage: ~150s = 0.04%)

**Result**: $0.00/month for your traffic volume

### Optional Cloud Scheduler
- **Cost**: ~$0.10/month
- **Benefit**: Keeps service warm for faster responses

## Troubleshooting

### Model not found error
```bash
# Train the model first
make train
```

### Cold starts too slow
```bash
# Set up keep-warm scheduler
make setup-scheduler
```

### Out of memory
```bash
# Increase memory in deployment
gcloud run services update invoice-classifier --memory 1Gi
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
