# Using Invoice Classifier as a Sidecar

Integration guide for running the invoice classifier as a sidecar container in your main application.

## Overview

The classifier container ships **without model files**. You provide trained models at runtime via volume mounts (local) or GCS bucket mounts (Cloud Run).

## Required Model Files

Place these files in a `ml-models/` directory in your main project:

```
ml-models/
  invoice_classifier.joblib       # Category model (~12MB)
  invoice_tag_classifier.joblib   # Tag model (~6MB)
```

These files are produced by `make train` in the classifier repo.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_TOKEN` | Yes | Bearer token for prediction endpoints |
| `MODEL_DIR` | No | Override model directory (default: `/app/models`) |
| `MODEL_VERSION` | No | Version string returned in responses (default: `1.0.0`) |
| `LOG_LEVEL` | No | Logging level (default: `info`) |
| `RATE_LIMIT_RPM` | No | Max requests per minute per IP (default: `60`) |

## Local Development (docker-compose)

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - CLASSIFIER_URL=http://classifier:8080
    depends_on:
      classifier:
        condition: service_healthy

  classifier:
    image: ghcr.io/your-org/invoice-classifier:latest
    # Or build from source:
    # build:
    #   context: ./path-to-classifier-repo
    volumes:
      - ./ml-models:/app/models:ro
    environment:
      - API_TOKEN=${CLASSIFIER_API_TOKEN}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 10s
      timeout: 3s
      start_period: 10s
      retries: 3
```

Then from your app, call the classifier at `http://classifier:8080`.

## Google Cloud Run (with GCS model storage)

### 1. Upload models to GCS

```bash
# Create a bucket for models (one-time)
gsutil mb -l europe-west1 gs://YOUR_PROJECT-ml-models

# Upload trained models
gsutil cp ml-models/invoice_classifier.joblib gs://YOUR_PROJECT-ml-models/
gsutil cp ml-models/invoice_tag_classifier.joblib gs://YOUR_PROJECT-ml-models/
```

### 2. Deploy with GCS volume mount

```bash
gcloud run deploy invoice-classifier \
  --image ghcr.io/your-org/invoice-classifier:latest \
  --region europe-west1 \
  --platform managed \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --cpu-boost \
  --timeout 60 \
  --port 8080 \
  --no-allow-unauthenticated \
  --set-env-vars "API_TOKEN=your-secret-token" \
  --add-volume name=models,type=cloud-storage,bucket=YOUR_PROJECT-ml-models \
  --add-volume-mount volume=models,mount-path=/app/models
```

### 3. Alternative: service.yaml

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: invoice-classifier
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 60
      containers:
        - image: ghcr.io/your-org/invoice-classifier:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              memory: 1Gi
              cpu: "1"
          env:
            - name: API_TOKEN
              valueFrom:
                secretKeyRef:
                  key: latest
                  name: classifier-api-token
            - name: LOG_LEVEL
              value: info
          volumeMounts:
            - name: models
              mountPath: /app/models
              readOnly: true
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            periodSeconds: 30
      volumes:
        - name: models
          csi:
            driver: gcsfuse.run.googleapis.com
            volumeAttributes:
              bucketName: YOUR_PROJECT-ml-models
              mountOptions: "implicit-dirs"
```

Deploy with:

```bash
# Create secret for API token (one-time)
echo -n "your-secret-token" | gcloud secrets create classifier-api-token --data-file=-

# Grant secret access to Cloud Run service account
gcloud secrets add-iam-policy-binding classifier-api-token \
  --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy
gcloud run services replace service.yaml --region europe-west1
```

## API Usage

All prediction endpoints require `Authorization: Bearer <API_TOKEN>` header.

### Predict category

```bash
curl -X POST http://classifier:8080/predict/category \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLASSIFIER_API_TOKEN" \
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

### Predict tag

```bash
curl -X POST http://classifier:8080/predict/tag \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLASSIFIER_API_TOKEN" \
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

### Health check

```bash
curl http://classifier:8080/health
```

## Keep-Warm (recommended)

With min-instances=0, add a keep-alive ping from your main app to avoid cold starts:

```python
# In your main app (example with httpx)
async def keep_classifier_warm():
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            await asyncio.sleep(240)  # every 4 minutes
            try:
                await client.get(f"{CLASSIFIER_URL}/health")
            except Exception:
                pass
```

## Updating Models

When you retrain models:

**Local**: Just replace files in `ml-models/` and restart the container.

**Cloud Run**: Upload new files to GCS and redeploy (or restart) the service:

```bash
gsutil cp ml-models/*.joblib gs://YOUR_PROJECT-ml-models/
gcloud run services update invoice-classifier --region europe-west1
```
