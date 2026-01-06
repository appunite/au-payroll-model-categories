# API Testing Examples

This directory contains example invoice data and testing utilities.

## Example Files

- `invoice_office_rent.json` - Office rent expense
- `invoice_utilities.json` - Utility bill (electricity)
- `invoice_software.json` - Software subscription

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
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @examples/invoice_office_rent.json
```

### Using httpie (if installed)

```bash
http POST http://localhost:8080/predict < examples/invoice_office_rent.json
```

### Using Python

```python
import requests

with open('examples/invoice_office_rent.json') as f:
    data = json.load(f)

response = requests.post('http://localhost:8080/predict', json=data)
print(response.json())
```

## Creating Custom Test Cases

Create a new JSON file with the invoice structure:

```json
{
  "entityId": "uuid",
  "ownerId": "uuid",
  "netPrice": 100.0,
  "grossPrice": 123.0,
  "currency": "PLN",
  "title_normalized": "description",
  "tin": "tax_id",
  "issueDate": "YYYY-MM-DD"
}
```

Then test it:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @your_invoice.json
```
