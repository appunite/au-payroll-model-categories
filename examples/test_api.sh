#!/bin/bash
# Test script for invoice classifier API
# Usage: ./test_api.sh [base_url]
# Example: ./test_api.sh http://localhost:8080
# Example: ./test_api.sh https://your-service.run.app

set -e

BASE_URL=${1:-http://localhost:8080}
EXAMPLES_DIR="$(dirname "$0")"

echo "Testing Invoice Classifier API at: $BASE_URL"
echo "================================================"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test health endpoint
echo -e "\n${BLUE}1. Testing /health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
echo "$HEALTH_RESPONSE" | python3 -m json.tool
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi

# Test root endpoint
echo -e "\n${BLUE}2. Testing / endpoint...${NC}"
ROOT_RESPONSE=$(curl -s "$BASE_URL/")
echo "$ROOT_RESPONSE" | python3 -m json.tool
if echo "$ROOT_RESPONSE" | grep -q "Invoice Classifier"; then
    echo -e "${GREEN}✓ Root endpoint passed${NC}"
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
    exit 1
fi

# Test prediction with different invoice types
test_prediction() {
    local file=$1
    local name=$(basename "$file" .json)

    echo -e "\n${BLUE}3. Testing prediction: $name${NC}"
    echo "Request:"
    cat "$file" | python3 -m json.tool

    echo -e "\nResponse:"
    PREDICTION=$(curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: application/json" \
        -d @"$file")

    echo "$PREDICTION" | python3 -m json.tool

    if echo "$PREDICTION" | grep -q "probabilities"; then
        echo -e "${GREEN}✓ Prediction successful${NC}"

        # Extract top category
        TOP_CATEGORY=$(echo "$PREDICTION" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['top_category'])")
        TOP_PROB=$(echo "$PREDICTION" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['top_probability']:.2%}\")")
        echo -e "  Top prediction: ${GREEN}$TOP_CATEGORY${NC} (${GREEN}$TOP_PROB${NC})"
    else
        echo -e "${RED}✗ Prediction failed${NC}"
        return 1
    fi
}

# Test each example file
for example_file in "$EXAMPLES_DIR"/invoice_*.json; do
    if [ -f "$example_file" ]; then
        test_prediction "$example_file"
    fi
done

# Test error handling
echo -e "\n${BLUE}4. Testing error handling (invalid data)...${NC}"
INVALID_REQUEST='{"entityId":"test","ownerId":"test","netPrice":-100,"grossPrice":100,"currency":"PLN","title_normalized":"test","issueDate":"2024-01-01"}'
ERROR_RESPONSE=$(curl -s -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$INVALID_REQUEST")

if echo "$ERROR_RESPONSE" | grep -q "detail"; then
    echo -e "${GREEN}✓ Error handling works${NC}"
    echo "$ERROR_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}✗ Error handling failed${NC}"
fi

echo -e "\n${GREEN}================================================"
echo "All tests completed!"
echo "================================================${NC}"
