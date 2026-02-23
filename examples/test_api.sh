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

# Test category prediction with different invoice types
test_category_prediction() {
    local file=$1
    local name
    name=$(basename "$file" .json)

    echo -e "\n${BLUE}3. Testing category prediction: $name${NC}"
    echo "Request:"
    cat "$file" | python3 -m json.tool

    echo -e "\nResponse:"
    local PREDICTION
    PREDICTION=$(curl -s -X POST "$BASE_URL/predict/category" \
        -H "Content-Type: application/json" \
        -d @"$file")

    echo "$PREDICTION" | python3 -m json.tool

    if echo "$PREDICTION" | grep -q "probabilities"; then
        echo -e "${GREEN}✓ Category prediction successful${NC}"

        # Extract top category
        local TOP_CATEGORY TOP_PROB
        TOP_CATEGORY=$(echo "$PREDICTION" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['top_category'])")
        TOP_PROB=$(echo "$PREDICTION" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['top_probability']:.2%}\")")
        echo -e "  Top prediction: ${GREEN}$TOP_CATEGORY${NC} (${GREEN}$TOP_PROB${NC})"
    else
        echo -e "${RED}✗ Category prediction failed${NC}"
        return 1
    fi
}

# Test tag prediction with different invoice types
test_tag_prediction() {
    local file=$1
    local name
    name=$(basename "$file" .json)

    echo -e "\n${BLUE}4. Testing tag prediction: $name${NC}"
    echo "Request:"
    cat "$file" | python3 -m json.tool

    echo -e "\nResponse:"
    local PREDICTION
    PREDICTION=$(curl -s -X POST "$BASE_URL/predict/tag" \
        -H "Content-Type: application/json" \
        -d @"$file")

    echo "$PREDICTION" | python3 -m json.tool

    if echo "$PREDICTION" | grep -q "probabilities"; then
        echo -e "${GREEN}✓ Tag prediction successful${NC}"

        # Extract top tag
        local TOP_TAG TOP_PROB
        TOP_TAG=$(echo "$PREDICTION" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['top_tag'])")
        TOP_PROB=$(echo "$PREDICTION" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['top_probability']:.2%}\")")
        echo -e "  Top prediction: ${GREEN}$TOP_TAG${NC} (${GREEN}$TOP_PROB${NC})"
    else
        echo -e "${RED}✗ Tag prediction failed${NC}"
        return 1
    fi
}

FAILED=0

# Test each example file
for example_file in "$EXAMPLES_DIR"/invoice_*.json; do
    if [ -f "$example_file" ]; then
        test_category_prediction "$example_file" || FAILED=1
        test_tag_prediction "$example_file" || FAILED=1
    fi
done

# Test error handling
echo -e "\n${BLUE}5. Testing error handling (invalid data)...${NC}"
INVALID_REQUEST='{"entity_id":"test","owner_id":"test","net_price":-100,"gross_price":100,"currency":"PLN","invoice_title":"test","issue_date":"2024-01-01"}'

echo -e "\n${BLUE}5a. Invalid data → /predict/category${NC}"
ERROR_RESPONSE=$(curl -s -X POST "$BASE_URL/predict/category" \
    -H "Content-Type: application/json" \
    -d "$INVALID_REQUEST")

if echo "$ERROR_RESPONSE" | grep -q "detail"; then
    echo -e "${GREEN}✓ Category error handling works${NC}"
    echo "$ERROR_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}✗ Category error handling failed${NC}"
    FAILED=1
fi

echo -e "\n${BLUE}5b. Invalid data → /predict/tag${NC}"
TAG_ERROR_RESPONSE=$(curl -s -X POST "$BASE_URL/predict/tag" \
    -H "Content-Type: application/json" \
    -d "$INVALID_REQUEST")

if echo "$TAG_ERROR_RESPONSE" | grep -q "detail"; then
    echo -e "${GREEN}✓ Tag error handling works${NC}"
    echo "$TAG_ERROR_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}✗ Tag error handling failed${NC}"
    FAILED=1
fi

echo -e "\n${GREEN}================================================"
echo "All tests completed!"
echo "================================================${NC}"

exit $FAILED
