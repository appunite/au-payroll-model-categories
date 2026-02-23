# API Examples & Integration Guide

This directory contains example invoice data, API response examples, and client implementations.

## Files

### JSON Examples

- `request.json` - Example invoice request payload
- `category_response.json` - Example category prediction response (36 categories)
- `tag_response.json` - Example tag prediction response (17 tags)
- `invoice_office_rent.json` - Office rent expense example
- `invoice_utilities.json` - Utility bill example
- `invoice_software.json` - Software subscription example

### Documentation

- `api_responses.md` - Complete API documentation with TypeScript, Python, and cURL examples

### Testing Scripts

- `test_api.sh` - Bash script for testing API endpoints

## Quick Test

### Test Local API

```bash
# Start the API (in another terminal)
make run

# Run tests
./examples/test_api.sh
```

### Test Deployed API

```bash
./examples/test_api.sh https://your-service.run.app
```

## Manual Testing

### Using curl

```bash
# Category prediction
curl -X POST http://localhost:8080/predict/category \
  -H "Content-Type: application/json" \
  -d @examples/invoice_office_rent.json

# Tag prediction
curl -X POST http://localhost:8080/predict/tag \
  -H "Content-Type: application/json" \
  -d @examples/invoice_office_rent.json
```

### Using httpie (if installed)

```bash
# Category prediction
http POST http://localhost:8080/predict/category < examples/invoice_office_rent.json

# Tag prediction
http POST http://localhost:8080/predict/tag < examples/invoice_office_rent.json
```

### Using Python

```python
import json
import requests

with open('examples/invoice_office_rent.json') as f:
    data = json.load(f)

# Category prediction
category = requests.post('http://localhost:8080/predict/category', json=data)
print(category.json())

# Tag prediction
tag = requests.post('http://localhost:8080/predict/tag', json=data)
print(tag.json())
```

## Creating Custom Test Cases

Create a new JSON file with the invoice structure:

```json
{
  "entity_id": "uuid",
  "owner_id": "uuid",
  "net_price": 100.0,
  "gross_price": 123.0,
  "currency": "PLN",
  "invoice_title": "description",
  "tin": "tax_id",
  "issue_date": "YYYY-MM-DD"
}
```

Then test it:

```bash
# Category prediction
curl -X POST http://localhost:8080/predict/category \
  -H "Content-Type: application/json" \
  -d @your_invoice.json

# Tag prediction
curl -X POST http://localhost:8080/predict/tag \
  -H "Content-Type: application/json" \
  -d @your_invoice.json
```
