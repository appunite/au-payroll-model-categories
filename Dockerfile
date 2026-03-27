# Multi-stage build for minimal image size and fast cold starts
# Using Python 3.11 slim for smaller base image

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency specifications
COPY pyproject.toml .

# Install dependencies to a virtual environment
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml

# Stage 2: Runtime image (minimal)
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Reduce Python memory overhead
    PYTHONHASHSEED=random \
    # Disable pip version check for faster startup
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy application code
COPY src/ ./src/

# Create models directory (models are mounted at runtime via Docker volumes)
RUN mkdir -p ./models

# Pre-compile .pyc files to avoid parsing overhead on cold start
RUN python -m compileall -q /opt/venv/lib/ ./src/ 2>/dev/null || true

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port (Cloud Run uses PORT env var, defaults to 8080)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=2 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application
# Using uvicorn with single worker for minimal memory footprint
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
