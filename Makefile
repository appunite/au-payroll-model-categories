.PHONY: help install fetch-data analyze-data train run test docker-build docker-run deploy clean format lint lint-fix

# Default target
help:
	@echo "Invoice Classifier - Available Commands:"
	@echo ""
	@echo "  make install       - Install dependencies using uv"
	@echo "  make fetch-data    - Fetch training data from PostgreSQL"
	@echo "  make analyze-data  - Analyze data distribution and get recommendations"
	@echo "  make train         - Train the ML model"
	@echo "  make run           - Run the API locally"
	@echo "  make test          - Run tests"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make lint-fix      - Lint and auto-fix issues"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container locally"
	@echo "  make deploy        - Deploy to Google Cloud Run"
	@echo "  make clean         - Remove generated files"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies with uv..."
	uv venv
	uv pip install -r pyproject.toml
	@echo "✓ Dependencies installed!"
	@echo "Activate venv with: source .venv/bin/activate"

# Fetch training data from PostgreSQL
fetch-data:
	@echo "Fetching training data from PostgreSQL..."
	@if [ ! -f .env ]; then \
		echo "ERROR: .env file not found."; \
		echo "Please copy .env.example to .env and fill in your database credentials:"; \
		echo "  cp .env.example .env"; \
		echo "  # Edit .env with your database credentials"; \
		exit 1; \
	fi
	uv run python src/fetch_training_data.py
	@echo "✓ Data fetched successfully!"

# Test database connection
test-db:
	@echo "Testing database connection..."
	uv run python src/fetch_training_data.py --dry-run

# Analyze data distribution
analyze-data:
	@echo "Analyzing data distribution..."
	@if [ ! -f data/invoices_training_data.csv ]; then \
		echo "ERROR: Training data not found."; \
		echo "Please fetch data first: make fetch-data"; \
		exit 1; \
	fi
	uv run python src/analyze_data.py

# Train model
train:
	@echo "Training invoice classifier..."
	@if [ ! -f data/invoices_training_data.csv ]; then \
		echo "ERROR: Training data not found at data/invoices_training_data.csv"; \
		echo "Please fetch training data first:"; \
		echo "  make fetch-data"; \
		exit 1; \
	fi
	uv run python src/train_model.py
	@echo "✓ Training complete!"

# Run API locally
run:
	@echo "Starting API server..."
	@if [ ! -f models/invoice_classifier.joblib ]; then \
		echo "ERROR: Model not found. Please run 'make train' first."; \
		exit 1; \
	fi
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8080

# Run tests
test:
	@echo "Running tests..."
	uv run pytest tests/ -v

# Test single prediction locally
test-predict:
	@echo "Testing prediction locally..."
	uv run python src/predict.py

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	@if [ ! -f models/invoice_classifier.joblib ]; then \
		echo "ERROR: Model not found. Please run 'make train' first."; \
		exit 1; \
	fi
	docker build -t invoice-classifier:latest .
	@echo "✓ Docker image built!"
	@echo "Image size:"
	@docker images invoice-classifier:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Run Docker container locally
docker-run:
	@echo "Running Docker container..."
	docker run --rm -p 8080:8080 invoice-classifier:latest

# Deploy to Google Cloud Run
deploy:
	@echo "Deploying to Google Cloud Run..."
	@if [ ! -f models/invoice_classifier.joblib ]; then \
		echo "ERROR: Model not found. Please run 'make train' first."; \
		exit 1; \
	fi
	@read -p "Enter Cloud Run service name [invoice-classifier]: " SERVICE_NAME; \
	SERVICE_NAME=$${SERVICE_NAME:-invoice-classifier}; \
	read -p "Enter GCP region [us-central1]: " REGION; \
	REGION=$${REGION:-us-central1}; \
	echo "Deploying $$SERVICE_NAME to $$REGION..."; \
	gcloud run deploy $$SERVICE_NAME \
		--source . \
		--region $$REGION \
		--platform managed \
		--allow-unauthenticated \
		--memory 512Mi \
		--cpu 1 \
		--max-instances 10 \
		--min-instances 0 \
		--cpu-boost \
		--timeout 60 \
		--port 8080
	@echo "✓ Deployment complete!"

# Set up Cloud Scheduler for keep-warm pings
setup-scheduler:
	@echo "Setting up Cloud Scheduler to keep service warm..."
	@read -p "Enter Cloud Run service URL: " SERVICE_URL; \
	read -p "Enter GCP region [us-central1]: " REGION; \
	REGION=$${REGION:-us-central1}; \
	gcloud scheduler jobs create http keep-invoice-classifier-warm \
		--schedule="*/5 * * * *" \
		--uri="$$SERVICE_URL/health" \
		--http-method=GET \
		--location=$$REGION \
		--description="Keep invoice classifier warm to avoid cold starts"
	@echo "✓ Scheduler created! Service will be pinged every 5 minutes."

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .venv/
	@echo "✓ Cleaned!"

# Format code with ruff
format:
	uv run ruff format src/ tests/

# Lint code with ruff
lint:
	uv run ruff check src/ tests/

# Lint and fix
lint-fix:
	uv run ruff check --fix src/ tests/
