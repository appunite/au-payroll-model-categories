# Invoice Category & Tag Prediction - Implementation Summary

## Overview

The codebase has been updated to support **two independent models**:
1. **Category Model** - Predicts expense categories (e.g., `office:rent`, `office:utilities`)
2. **Tag Model** - Predicts expense tags (e.g., `visual-panda`, `benefit-training`, `accounting`)

Both models use the same input features but are trained on different target variables.

## Architecture Changes

### File Structure

```
src/
├── config.py                    # Updated: Added TAG_MODEL_PATH
├── preprocessing.py             # NEW: Shared preprocessing utilities
├── train_model_category.py      # Refactored: Category model training
├── train_model_tag.py           # NEW: Tag model training
├── predict.py                   # Updated: Both category & tag prediction
└── main.py                      # Updated: Separate API endpoints

queries/
└── fetch_tag_training_data.sql  # NEW: SQL query for tag training data

models/
├── invoice_classifier.joblib          # Category model
├── invoice_tag_classifier.joblib      # Tag model
├── category_model_metrics.json        # Category metrics
└── tag_model_metrics.json             # Tag metrics
```

### Key Design Decisions

1. **Separate Models**: Two independent LightGBM models for better accuracy
2. **Shared Preprocessing**: `preprocessing.py` contains common feature engineering logic
3. **Dual Model Caching**: Both models loaded on startup for fast inference
4. **Separate Endpoints**: `/predict/category` and `/predict/tag`

## Training the Models

### 1. Category Model

**SQL Query**: Use your existing query for `expenseCategory` data

**Training**:
```bash
# Export training data from database
# Save as: data/invoices_training_data.csv

# Train model
python -m src.train_model_category

# Or if using a different CSV file:
python -m src.train_model_category data/my_category_data.csv
```

**Output**:
- Model: `models/invoice_classifier.joblib`
- Metrics: `models/category_model_metrics.json`

### 2. Tag Model

**SQL Query**: Use the provided query in `queries/fetch_tag_training_data.sql`

**Training**:
```bash
# Export training data using queries/fetch_tag_training_data.sql
# Save as: data/invoices_tag_training_data.csv

# Train model
python -m src.train_model_tag

# Or if using a different CSV file:
python -m src.train_model_tag data/my_tag_data.csv
```

**Output**:
- Model: `models/invoice_tag_classifier.joblib`
- Metrics: `models/tag_model_metrics.json`

## API Endpoints

### 1. Predict Category

**Endpoint**: `POST /predict/category`

**Request**:
```json
{
  "entityId": "00000000-0000-0000-0000-000000000001",
  "ownerId": "00000000-0000-0000-0000-000000000002",
  "netPrice": 2500.0,
  "grossPrice": 3075.0,
  "currency": "PLN",
  "invoice_title": "Adobe Systems Software Ireland Ltd",
  "tin": "1234567890",
  "issueDate": "2024-08-29"
}
```

**Response**:
```json
{
  "probabilities": {
    "office:software": 0.85,
    "office:subscriptions": 0.10,
    "others:other": 0.05
  },
  "top_category": "office:software",
  "top_probability": 0.85,
  "model_version": "1.0.0"
}
```

### 2. Predict Tag

**Endpoint**: `POST /predict/tag`

**Request**: Same as category prediction

**Response**:
```json
{
  "probabilities": {
    "visual-panda": 0.75,
    "referral-fee": 0.15,
    "accounting": 0.10
  },
  "top_category": "visual-panda",
  "top_probability": 0.75,
  "model_version": "1.0.0"
}
```

### 3. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2024-01-07T12:00:00"
}
```

Note: Health check returns `"healthy"` only if **both** models are loaded.

## Shared Preprocessing

Both models use identical feature engineering:

**Numerical Features**:
- `netPrice`
- `VAT_Amount` (derived: `grossPrice - netPrice`)
- `VAT_Rate` (derived: `grossPrice / netPrice - 1`)

**Categorical Features**:
- `entityId`
- `ownerId`
- `currency`
- `tin`

**Datetime Features** (derived from `issueDate`):
- `issueYear`
- `issueMonth`
- `issueDay`

**Text Feature**:
- `invoice_title` (processed with TF-IDF vectorization)

## Testing Locally

### Test Prediction Functions

```bash
# Test both category and tag prediction
python -m src.predict
```

This will run example predictions for both models and display results.

### Test API Server

```bash
# Start server
make run
# or
python -m src.main

# Test category prediction
curl -X POST http://localhost:8080/predict/category \
  -H "Content-Type: application/json" \
  -d '{
    "entityId": "00000000-0000-0000-0000-000000000001",
    "ownerId": "00000000-0000-0000-0000-000000000002",
    "netPrice": 2500.0,
    "grossPrice": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issueDate": "2024-08-29"
  }'

# Test tag prediction
curl -X POST http://localhost:8080/predict/tag \
  -H "Content-Type: application/json" \
  -d '{
    "entityId": "00000000-0000-0000-0000-000000000001",
    "ownerId": "00000000-0000-0000-0000-000000000002",
    "netPrice": 2500.0,
    "grossPrice": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issueDate": "2024-08-29"
  }'
```

## Deployment

The deployment process remains the same, but now both models will be loaded:

1. **Build Docker Image**:
   ```bash
   docker build -t invoice-classifier .
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy invoice-classifier \
     --source . \
     --region us-central1 \
     --memory 512Mi
   ```

3. **Cold Start Optimization**:
   - Both models are loaded during startup
   - Global model caching ensures single load per container
   - Total memory footprint: ~400-500MB (within 512Mi limit)

## Migration Notes

### Breaking Changes

- **Old endpoint** `/predict` has been **replaced** with `/predict/category` and `/predict/tag`
- No backward compatibility maintained (as requested)

### What to Update

1. **Client Code**: Update API calls from `/predict` to `/predict/category` or `/predict/tag`
2. **Training Scripts**: Use separate scripts for category and tag training
3. **Model Files**: Ensure both model files are deployed

## Performance Considerations

### Model Sizes

- **Category Model**: ~50-200MB (depends on training data)
- **Tag Model**: ~50-200MB (depends on training data)
- **Total**: ~100-400MB for both models

### Inference Latency

- **Cold Start**: 3-5 seconds (loads both models)
- **Warm Request**: <200ms per prediction
- **Both Predictions**: Can be called sequentially, total <400ms

### Memory Usage

- **Total Budget**: 512Mi
- **Python Runtime**: ~50MB
- **Dependencies**: ~150MB
- **Both Models**: ~100-400MB
- **Request Handling**: ~50MB
- **Expected Total**: ~350-650MB

**Note**: If you exceed 512Mi, increase Cloud Run memory allocation to 1Gi.

## Tag Query Details

The tag training query differs from category in two key ways:

1. **Filters for tagged invoices**: `expenseTags <> '{}' AND expenseTags IS NOT NULL`
2. **Unnests tags array**: `LATERAL unnest(main."expenseTags") AS tag`
3. **Filters specific tags**: Only includes predefined tags like `'visual-panda'`, `'benefit-training'`, etc.

This creates one training row per tag, allowing the model to learn tag-specific patterns.

## Next Steps

1. **Export training data** using the provided SQL queries
2. **Train both models** using the training scripts
3. **Test locally** using `python -m src.predict` and the API endpoints
4. **Deploy** to Cloud Run with both model files
5. **Update client code** to use new endpoints

## Questions?

- Check model metrics in `models/category_model_metrics.json` and `models/tag_model_metrics.json`
- Review training logs for accuracy, precision, recall, and F1 scores
- Monitor Cloud Run logs for inference latency and errors
