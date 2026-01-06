# Invoice Classifier - Best Practices & Architecture

This document outlines the architectural decisions, best practices, and optimizations implemented in this invoice classification system.

## Architecture Overview

### System Design

```
Training Pipeline:
SQL Database → CSV Export → train_model.py → LightGBM Model → joblib file

Inference Pipeline:
HTTP Request → FastAPI → predict.py → Model (cached) → JSON Response
```

### Technology Stack

- **ML Framework**: LightGBM (faster & lighter than sklearn GradientBoosting)
- **API Framework**: FastAPI (async-capable, auto-docs, type validation)
- **Server**: Uvicorn (ASGI server)
- **Validation**: Pydantic (request/response validation)
- **Deployment**: Google Cloud Run (serverless, auto-scaling)
- **Package Manager**: uv (faster than pip)

## Performance Optimizations

### 1. Cold Start Optimization

**Problem**: Serverless functions suffer from cold starts (3-10 seconds)

**Solutions Implemented**:

#### a) Minimal Docker Image
```dockerfile
# Multi-stage build
FROM python:3.11-slim AS builder
# Install dependencies in isolated stage

FROM python:3.11-slim
# Copy only runtime dependencies
```

**Impact**: Reduces image size from ~1GB to ~300-500MB

#### b) Global Model Caching
```python
# predict.py
_model_cache = None  # Global variable

def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache
```

**Impact**: Model loaded once per container instance, not per request

#### c) Startup Loading
```python
# main.py
@app.on_event("startup")
async def startup_event():
    load_model()  # Pre-load during startup
```

**Impact**: First request doesn't pay model loading cost

#### d) LightGBM vs sklearn
- **sklearn GradientBoosting**: Large model file, slower inference
- **LightGBM**: Smaller models, 2-3x faster inference

**Impact**: 30-50% faster cold starts

#### e) Single Worker Configuration
```bash
CMD ["uvicorn", "src.main:app", "--workers", "1"]
```

**Impact**: Minimal memory footprint, faster initialization

### 2. Inference Performance

**Target**: Sub-200ms for warm containers

**Optimizations**:

1. **Feature preprocessing cached in pipeline**
   ```python
   pipeline = Pipeline([
       ('preprocessor', preprocessor),  # Cached transformers
       ('model', model)
   ])
   ```

2. **Efficient data types**
   - Use `pd.DataFrame` for batch compatibility
   - Convert to native Python types for JSON serialization

3. **Single-threaded prediction**
   ```python
   LGBM_PARAMS = {
       'n_jobs': 1,  # Single thread for consistent latency
   }
   ```

### 3. Memory Optimization

**Memory Budget**: 512Mi for Cloud Run free tier

**Breakdown**:
- Python runtime: ~50MB
- Dependencies: ~150MB
- Model: ~50-200MB (depends on training data)
- Request handling: ~50MB
- **Total**: ~300-450MB

**Techniques**:
- Use `python:3.11-slim` (vs full Python image saves 500MB)
- Minimal dependencies (no unnecessary packages)
- Single worker (vs multiple workers)

## Model Training Best Practices

### 1. Feature Engineering

```python
# Derived features for better predictions
df['VAT_Amount'] = df['grossPrice'] - df['netPrice']
df['VAT_Rate'] = (df['grossPrice'] / df['netPrice']) - 1

# Temporal features
df['issueYear'] = df['issueDate'].dt.year
df['issueMonth'] = df['issueDate'].dt.month
df['issueDay'] = df['issueDate'].dt.day
```

**Why**: VAT patterns differ by category; temporal patterns exist in expenses

### 2. Preprocessing Pipeline

```python
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), NUMERICAL_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ('datetime', 'passthrough', DATETIME_FEATURES)
])
```

**Benefits**:
- Handles missing data automatically
- `handle_unknown='ignore'` prevents errors on new categories
- Entire pipeline serialized with model (no preprocessing drift)

### 3. Model Selection

**Why LightGBM over alternatives**:

| Model | Training Speed | Inference Speed | Model Size | Accuracy |
|-------|---------------|-----------------|------------|----------|
| Random Forest | Medium | Slow | Large | Good |
| sklearn GradientBoosting | Slow | Medium | Medium | Good |
| **LightGBM** | **Fast** | **Fast** | **Small** | **Best** |
| XGBoost | Fast | Fast | Medium | Best |

LightGBM chosen for:
- Fastest inference (critical for API latency)
- Smallest model size (faster cold starts)
- Best accuracy on tabular data

### 4. Evaluation Metrics

```python
metrics = {
    'cv_mean_accuracy': cross_val_score(...).mean(),
    'test_accuracy': accuracy_score(...),
    'test_precision': precision_score(...),
    'test_recall': recall_score(...),
    'test_f1': f1_score(...),
}
```

**Why all four metrics**:
- **Accuracy**: Overall correctness
- **Precision**: Important for high-stakes categories (avoid false positives)
- **Recall**: Catch all instances of important categories
- **F1**: Balanced metric for imbalanced classes

### 5. Cross-Validation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = cross_val_score(pipeline, X, y, cv=cv)
```

**Benefits**:
- Stratified ensures class balance in folds
- 5 folds is standard (good bias-variance tradeoff)
- Validates generalization before deployment

## API Design Best Practices

### 1. Request Validation

```python
class InvoiceRequest(BaseModel):
    entityId: str
    netPrice: float = Field(..., gt=0)  # Must be positive
    currency: str = Field(..., min_length=3, max_length=3)
    # ...
```

**Benefits**:
- Automatic validation (400 errors for bad data)
- Type safety
- Auto-generated OpenAPI docs

### 2. Error Handling

```python
try:
    probabilities = predict_expense_category(...)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except FileNotFoundError as e:
    raise HTTPException(status_code=503, detail="Model not available")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

**Why specific exceptions**:
- 400: Client error (bad input)
- 503: Service unavailable (model not loaded)
- 500: Server error (unexpected failure)

### 3. Health Checks

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }
```

**Uses**:
- Cloud Run health monitoring
- Keep-alive pings from Cloud Scheduler
- Debugging deployment issues

## Deployment Strategy

### 1. Cloud Run Configuration

```yaml
Memory: 512Mi          # Sufficient for model + runtime
CPU: 1                 # Standard for ML inference
Min instances: 0       # Scale to zero (free tier)
Max instances: 10      # Handle traffic spikes
CPU boost: enabled     # Faster cold starts
Timeout: 60s           # Enough for cold start
```

### 2. Keep-Warm Strategy

**Problem**: 20-50 requests/day means frequent cold starts

**Solution**: Cloud Scheduler pinging every 5 minutes

```bash
gcloud scheduler jobs create http keep-warm \
  --schedule="*/5 * * * *" \
  --uri="https://service.run.app/health"
```

**Math**:
- Container stays warm for ~15 minutes after last request
- Ping every 5 minutes = always warm
- Cost: ~$0.10/month (8,640 pings × $0.00001)

**Alternative**: Min instances = 1 costs ~$8-12/month

### 3. Environment Variables

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env locally
        case_sensitive=False       # LOG_LEVEL = log_level
    )
```

**Benefits**:
- Local development uses `.env` file
- Cloud Run uses environment variables
- Same code works in both environments

## Testing Strategy

### 1. Unit Tests

```python
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
```

**Coverage**:
- API endpoints (without model dependency)
- Request validation
- Error handling

### 2. Integration Tests

```python
@pytest.mark.skipif(not model_exists, reason="Requires trained model")
def test_predict():
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
```

**When to run**:
- After training new model
- Before deployment
- In CI/CD pipeline

### 3. Local Testing

```bash
# Test prediction logic directly
python src/predict.py

# Test API locally
make run
curl http://localhost:8080/predict -d @test_invoice.json
```

## Security Best Practices

### 1. Container Security

```dockerfile
# Non-root user
RUN useradd -m -u 1000 appuser
USER appuser
```

**Why**: Principle of least privilege

### 2. Input Validation

All inputs validated by Pydantic:
- Type checking
- Range validation (e.g., `netPrice > 0`)
- String length limits

### 3. Dependency Management

```toml
[project]
dependencies = [
    "fastapi>=0.109.0",  # Pinned major version
]
```

**Benefits**:
- Reproducible builds
- Security updates allowed (minor/patch)
- No breaking changes (major version pinned)

## Monitoring & Observability

### 1. Structured Logging

```python
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger.info(f"Predicting category for: {invoice.title_normalized}")
```

**Cloud Run Integration**: Logs automatically sent to Cloud Logging

### 2. Metrics to Monitor

- **Request latency** (p50, p95, p99)
- **Error rate** (4xx, 5xx)
- **Cold start frequency**
- **Memory usage**
- **Model accuracy** (track predictions over time)

### 3. Model Versioning

```python
MODEL_VERSION = "1.0.0"  # In config.py

# Returned in every prediction
{
  "model_version": "1.0.0",
  "probabilities": {...}
}
```

**Benefits**:
- Track which model version made each prediction
- A/B testing different models
- Rollback if new model underperforms

## Future Improvements

### 1. Model Improvements

- **Online learning**: Update model with user corrections
- **Active learning**: Request labels for low-confidence predictions
- **Feature store**: Track feature distributions over time
- **Model monitoring**: Detect data drift, concept drift

### 2. Performance Improvements

- **Model quantization**: Reduce model size by 50-75%
- **ONNX conversion**: Faster inference runtime
- **Batch prediction**: Amortize overhead across multiple requests
- **Caching**: Cache predictions for identical inputs

### 3. Operational Improvements

- **CI/CD pipeline**: Automated testing and deployment
- **Canary deployments**: Gradual rollout of new models
- **Blue-green deployments**: Zero-downtime updates
- **Automated retraining**: Periodic retraining with new data

## Cost Optimization

### Current Setup (Free Tier)

| Resource | Usage | Limit | Utilization |
|----------|-------|-------|-------------|
| Requests | 1,500/month | 2M/month | 0.075% |
| CPU | 300 vCPU-sec | 180k | 0.17% |
| Memory | 150 GiB-sec | 360k | 0.04% |

**Total cost**: $0.00/month

### With Keep-Warm

| Resource | Cost |
|----------|------|
| Cloud Scheduler | $0.10/month |
| Cloud Run (pings) | $0.00 (within free tier) |

**Total cost**: ~$0.10/month

### Scaling Considerations

If traffic grows to 1,000 requests/day (30k/month):

- Still within free tier (1.5% of request limit)
- May need min instances = 1 for consistent latency
- Cost would increase to ~$8-12/month

## Conclusion

This architecture balances:
- **Performance**: Fast inference (<200ms warm, 3-5s cold)
- **Cost**: Free tier for current traffic
- **Scalability**: Can handle 100x traffic increase
- **Maintainability**: Simple, clean codebase
- **Reliability**: Proper error handling, health checks

The key insight: For low-traffic ML APIs, **optimize for cold starts and stay in free tier** rather than over-engineering for scale you don't need yet.
