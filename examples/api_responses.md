# API Response Examples

## Request Validation Rules

All requests must comply with the following validation rules:

**Required Fields:**
- `entity_id` (string): Company/entity unique identifier
- `owner_id` (string): Invoice owner unique identifier
- `net_price` (float): Must be > 0
- `gross_price` (float): Must be > 0
- `currency` (string): Must be a valid ISO 4217 currency code (3 letters)
- `invoice_title` (string): Must not be empty
- `issue_date` (string): Must be in YYYY-MM-DD format

**Optional Fields:**
- `tin` (string or null): Tax identification number

**Currency Validation:**
- Must be exactly 3 characters
- Must be a valid ISO 4217 code (case-insensitive, will be normalized to uppercase)
- Supported currencies: PLN, USD, EUR, GBP, CHF, CZK, DKK, SEK, NOK, CAD, AUD, JPY, CNY, INR, BRL, MXN, ZAR, SGD, HKD, NZD, KRW, TRY, RUB, AED, SAR, THB, MYR, IDR, PHP, VND, ILS, RON, HUF, BGN, HRK, ISK

**Date Validation:**
- Format: YYYY-MM-DD (e.g., "2024-08-29")
- Must be a valid calendar date
- Must be after 2000-01-01
- Must not be in the future

---

## Category Prediction Response

**Endpoint**: `POST /predict/category`

**Request**:
```json
{
  "entity_id": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
  "owner_id": "e148cdec-d66d-11e9-8a40-47a686a82f23",
  "net_price": 2500.0,
  "gross_price": 3075.0,
  "currency": "PLN",
  "invoice_title": "Adobe Systems Software Ireland Ltd",
  "tin": "1234567890",
  "issue_date": "2024-08-29"
}
```

**Response**:
```json
{
  "probabilities": {
    "operations:design": 0.3724685157271892,
    "people:training": 0.11110586024300262,
    "marketing:services": 0.10573809683112458,
    "operations:services": 0.054385289940420245,
    "operations:administration": 0.04187419273042248,
    "operations:legal": 0.03641563053330512,
    "operations:essential": 0.0185045070549437,
    "operations:accountancy": 0.014867433002710206,
    "operations:infrastructure": 0.01429345832464022,
    "office:equipment": 0.014007244697640232,
    "people:events": 0.012677442867500355,
    "marketing:ads": 0.010567003488547031,
    "people:delivery": 0.010169766957733327,
    "people:gifts": 0.010026029809406244,
    "people:outings": 0.009575060948068863,
    "people:transportation": 0.008535418928711336,
    "people:books": 0.008520346085982847,
    "people:supply": 0.00836768072974473,
    "sales:services": 0.008218172441641927,
    "people:benefits": 0.008024281204349212,
    "people:employer-branding": 0.007936624257870858,
    "assets:devices": 0.007806196994462346,
    "operations:ai": 0.00768491029914507,
    "people:apartments": 0.007680143091026831,
    "marketing:conferences": 0.007584354170726702,
    "people:patronage": 0.007574386631546807,
    "others:ska-fleet": 0.007573112943689334,
    "recruitment:ads": 0.007570994397706795,
    "operations:exceptional": 0.007568665398190752,
    "people:newsletter": 0.007561195228708302,
    "recruitment:services": 0.007557790820809793,
    "assets:repairs": 0.007556230110011082,
    "office:rent-and-administration": 0.0075493570812627826,
    "assets:accessories": 0.007524587352628398,
    "office:maintenance": 0.0074943912911729395,
    "people:services": 0.007435627383956765
  },
  "top_category": "operations:design",
  "top_probability": 0.3724685157271892,
  "model_version": "1.0.0"
}
```

**Response Fields**:
- `probabilities`: Object with all 36 categories and their probability scores (0-1)
- `top_category`: The category with the highest probability
- `top_probability`: The probability score for the top category
- `model_version`: Version of the model used for prediction

---

## Tag Prediction Response

**Endpoint**: `POST /predict/tag`

**Request**:
```json
{
  "entity_id": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
  "owner_id": "e148cdec-d66d-11e9-8a40-47a686a82f23",
  "net_price": 2500.0,
  "gross_price": 3075.0,
  "currency": "PLN",
  "invoice_title": "Adobe Systems Software Ireland Ltd",
  "tin": "1234567890",
  "issue_date": "2024-08-29"
}
```

**Response**:
```json
{
  "probabilities": {
    "legal-advice": 0.45657710684807895,
    "benefit-training": 0.3922044384455597,
    "esop": 0.027944264711362753,
    "dashbit-jose-valim": 0.018777091912633007,
    "referral-fee": 0.018777091912633007,
    "benefit-outing": 0.007466479835417441,
    "benefit-psychologist": 0.0071404727621614695,
    "benefit-medical-care": 0.007128616812919834,
    "benefit-english": 0.007128542206759688,
    "benefit-multisport": 0.0071249544787428355,
    "accounting": 0.0071216214807763144,
    "benefit-books-formula": 0.007121277432091303,
    "benefit-insurance": 0.007119689431371918,
    "visual-panda": 0.00711867285359645,
    "benefit-computer-formula": 0.007099851355083948,
    "BHP": 0.00709888011327466,
    "benefit-apartments": 0.00705094740753674
  },
  "top_category": "legal-advice",
  "top_probability": 0.45657710684807895,
  "model_version": "1.0.0"
}
```

**Response Fields**:
- `probabilities`: Object with all 17 tags and their probability scores (0-1)
- `top_category`: The tag with the highest probability (note: field name is reused for consistency)
- `top_probability`: The probability score for the top tag
- `model_version`: Version of the model used for prediction

---

## All Available Categories (36)

```
- assets:accessories
- assets:devices
- assets:repairs
- marketing:ads
- marketing:conferences
- marketing:services
- office:equipment
- office:maintenance
- office:rent-and-administration
- operations:accountancy
- operations:administration
- operations:ai
- operations:design
- operations:essential
- operations:exceptional
- operations:infrastructure
- operations:legal
- operations:services
- others:ska-fleet
- people:apartments
- people:benefits
- people:books
- people:delivery
- people:employer-branding
- people:events
- people:gifts
- people:newsletter
- people:outings
- people:patronage
- people:services
- people:supply
- people:training
- people:transportation
- recruitment:ads
- recruitment:services
- sales:services
```

## All Available Tags (17)

```
- BHP
- accounting
- benefit-apartments
- benefit-books-formula
- benefit-computer-formula
- benefit-english
- benefit-insurance
- benefit-medical-care
- benefit-multisport
- benefit-outing
- benefit-psychologist
- benefit-training
- dashbit-jose-valim
- esop
- legal-advice
- referral-fee
- visual-panda
```

---

## Health Check Response

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-01-08T01:08:42.084177"
}
```

**Response Fields**:
- `status`: Either "healthy" or "unhealthy" (both models must be loaded for healthy)
- `model_loaded`: Boolean indicating if models are loaded
- `model_version`: Current model version
- `timestamp`: ISO 8601 timestamp of the response

---

## Root Endpoint Response

**Endpoint**: `GET /`

**Response**:
```json
{
  "name": "Invoice Classifier API",
  "version": "1.0.0",
  "endpoints": {
    "predict_category": "/predict/category",
    "predict_tag": "/predict/tag",
    "health": "/health",
    "docs": "/docs"
  }
}
```

---

## Error Responses

### 400 Bad Request (Validation Error)

**Invalid price (negative or zero):**
```json
{
  "detail": [
    {
      "loc": ["body", "net_price"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

**Invalid currency code:**
```json
{
  "detail": [
    {
      "loc": ["body", "currency"],
      "msg": "Value error, Invalid currency code 'XXX'. Must be a valid ISO 4217 code (e.g., PLN, USD, EUR, GBP)",
      "type": "value_error"
    }
  ]
}
```

**Invalid date format:**
```json
{
  "detail": [
    {
      "loc": ["body", "issue_date"],
      "msg": "Value error, Invalid date format '2024/08/29'. Must be YYYY-MM-DD (e.g., 2024-08-29)",
      "type": "value_error"
    }
  ]
}
```

**Future date:**
```json
{
  "detail": [
    {
      "loc": ["body", "issue_date"],
      "msg": "Value error, Date '2030-01-01' is in the future. Must not be later than today",
      "type": "value_error"
    }
  ]
}
```

**Date too old:**
```json
{
  "detail": [
    {
      "loc": ["body", "issue_date"],
      "msg": "Value error, Date '1999-12-31' is too old. Must be after 2000-01-01",
      "type": "value_error"
    }
  ]
}
```

### 503 Service Unavailable (Model Not Loaded)

```json
{
  "detail": "Category model not available"
}
```

or

```json
{
  "detail": "Tag model not available"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Prediction failed: <error message>"
}
```

---

## Usage Examples

### TypeScript/JavaScript

```typescript
interface InvoiceData {
  entity_id: string;
  owner_id: string;
  net_price: number;
  gross_price: number;
  currency: string;
  invoice_title: string;
  tin: string | null;
  issue_date: string; // YYYY-MM-DD
}

interface PredictionResponse {
  probabilities: Record<string, number>;
  top_category: string;
  top_probability: number;
  model_version: string;
}

async function predictCategory(invoice: InvoiceData): Promise<PredictionResponse> {
  const response = await fetch(
    'https://payroll-invoice-classifier-324047048236.us-central1.run.app/predict/category',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(invoice)
    }
  );

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }

  return response.json();
}

async function predictTag(invoice: InvoiceData): Promise<PredictionResponse> {
  const response = await fetch(
    'https://payroll-invoice-classifier-324047048236.us-central1.run.app/predict/tag',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(invoice)
    }
  );

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }

  return response.json();
}

// Usage
const invoice = {
  entity_id: "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
  owner_id: "e148cdec-d66d-11e9-8a40-47a686a82f23",
  net_price: 2500.0,
  gross_price: 3075.0,
  currency: "PLN",
  invoice_title: "Adobe Systems Software Ireland Ltd",
  tin: "1234567890",
  issue_date: "2024-08-29"
};

const categoryResult = await predictCategory(invoice);
console.log('Category:', categoryResult.top_category); // "operations:design"
console.log('Confidence:', categoryResult.top_probability); // 0.372

const tagResult = await predictTag(invoice);
console.log('Tag:', tagResult.top_category); // "legal-advice"
console.log('Confidence:', tagResult.top_probability); // 0.457
```

### Python

```python
import requests
from typing import Dict, Any

def predict_category(invoice: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        'https://payroll-invoice-classifier-324047048236.us-central1.run.app/predict/category',
        json=invoice
    )
    response.raise_for_status()
    return response.json()

def predict_tag(invoice: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        'https://payroll-invoice-classifier-324047048236.us-central1.run.app/predict/tag',
        json=invoice
    )
    response.raise_for_status()
    return response.json()

# Usage
invoice = {
    "entity_id": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
    "owner_id": "e148cdec-d66d-11e9-8a40-47a686a82f23",
    "net_price": 2500.0,
    "gross_price": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issue_date": "2024-08-29"
}

category_result = predict_category(invoice)
print(f"Category: {category_result['top_category']}")
print(f"Confidence: {category_result['top_probability']:.2%}")

tag_result = predict_tag(invoice)
print(f"Tag: {tag_result['top_category']}")
print(f"Confidence: {tag_result['top_probability']:.2%}")
```

### cURL

```bash
# Category prediction
curl -X POST https://payroll-invoice-classifier-324047048236.us-central1.run.app/predict/category \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
    "owner_id": "e148cdec-d66d-11e9-8a40-47a686a82f23",
    "net_price": 2500.0,
    "gross_price": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issue_date": "2024-08-29"
  }'

# Tag prediction
curl -X POST https://payroll-invoice-classifier-324047048236.us-central1.run.app/predict/tag \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "c2b6df6b-35e9-4120-9e7c-d20be39d7146",
    "owner_id": "e148cdec-d66d-11e9-8a40-47a686a82f23",
    "net_price": 2500.0,
    "gross_price": 3075.0,
    "currency": "PLN",
    "invoice_title": "Adobe Systems Software Ireland Ltd",
    "tin": "1234567890",
    "issue_date": "2024-08-29"
  }'
```
