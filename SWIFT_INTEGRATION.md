# Swift Integration Guide: Invoice Classifier on Cloud Run

This guide helps you integrate the payroll invoice classifier (category + tag prediction) into your Swift application.

## Overview

**Service:** Google Cloud Run
**Service URL:** `https://your-service.run.app`
**Framework:** FastAPI
**Response Format:** JSON (snake_case field names)
**Models:** Category (36 classes) + Tag (17 classes)

## API Endpoints

### Predict Invoice Category

**Endpoint:** `POST /predict/category`
**URL:** `https://your-service.run.app/predict/category`
**Content-Type:** `application/json`
**Authentication:** None (public endpoint)

### Predict Invoice Tag

**Endpoint:** `POST /predict/tag`
**URL:** `https://your-service.run.app/predict/tag`
**Content-Type:** `application/json`
**Authentication:** None (public endpoint)

### Request Payload (same for both endpoints)

```json
{
  "entity_id": "00000000-0000-0000-0000-000000000001",
  "owner_id": "00000000-0000-0000-0000-000000000002",
  "net_price": 2500.0,
  "gross_price": 3075.0,
  "currency": "PLN",
  "invoice_title": "Adobe Systems Software Ireland Ltd",
  "tin": "1234567890",
  "issue_date": "2024-08-29"
}
```

### Category Response Format

```json
{
  "probabilities": {
    "operations:design": 0.37,
    "people:training": 0.11,
    "marketing:services": 0.10,
    "...": "..."
  },
  "top_category": "operations:design",
  "top_probability": 0.37,
  "model_version": "1.0.0"
}
```

### Tag Response Format

```json
{
  "probabilities": {
    "legal-advice": 0.46,
    "benefit-training": 0.39,
    "esop": 0.03,
    "...": "..."
  },
  "top_tag": "legal-advice",
  "top_probability": 0.46,
  "model_version": "1.0.0"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_id` | String | Yes | Unique identifier for the company/entity |
| `owner_id` | String | Yes | Unique identifier for the invoice owner |
| `net_price` | Double | Yes | Net price excluding VAT (must be > 0) |
| `gross_price` | Double | Yes | Gross price including VAT (must be > 0) |
| `currency` | String | Yes | 3-letter currency code (PLN, USD, EUR, GBP) |
| `invoice_title` | String | Yes | Full invoice title/description (not just first word) |
| `tin` | String? | No | Tax identification number (can be null or empty string) |
| `issue_date` | String | Yes | Invoice issue date in YYYY-MM-DD format |

**Category Response Fields:**
- `probabilities`: Dictionary of all category names with their probability scores (0.0 to 1.0)
- `top_category`: The category with the highest probability
- `top_probability`: The probability score of the top prediction
- `model_version`: Model version used for prediction

**Tag Response Fields:**
- `probabilities`: Dictionary of all tag names with their probability scores (0.0 to 1.0)
- `top_tag`: The tag with the highest probability
- `top_probability`: The probability score of the top prediction
- `model_version`: Model version used for prediction

## Swift Implementation

### 1. Define Request/Response Models

```swift
import Foundation

// MARK: - Request Model
struct InvoiceClassificationRequest: Codable {
    let entityId: String
    let ownerId: String
    let netPrice: Double
    let grossPrice: Double
    let currency: String
    let invoiceTitle: String
    let tin: String?
    let issueDate: String

    enum CodingKeys: String, CodingKey {
        case entityId = "entity_id"
        case ownerId = "owner_id"
        case netPrice = "net_price"
        case grossPrice = "gross_price"
        case currency
        case invoiceTitle = "invoice_title"
        case tin
        case issueDate = "issue_date"
    }
}

// MARK: - Category Response Model
struct CategoryPredictionResponse: Codable {
    let probabilities: [String: Double]
    let topCategory: String
    let topProbability: Double
    let modelVersion: String

    enum CodingKeys: String, CodingKey {
        case probabilities
        case topCategory = "top_category"
        case topProbability = "top_probability"
        case modelVersion = "model_version"
    }
}

// MARK: - Tag Response Model
struct TagPredictionResponse: Codable {
    let probabilities: [String: Double]
    let topTag: String
    let topProbability: Double
    let modelVersion: String

    enum CodingKeys: String, CodingKey {
        case probabilities
        case topTag = "top_tag"
        case topProbability = "top_probability"
        case modelVersion = "model_version"
    }
}

// MARK: - Health Check Response
struct HealthResponse: Codable {
    let status: String
    let modelLoaded: Bool
    let modelVersion: String
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case status
        case modelLoaded = "model_loaded"
        case modelVersion = "model_version"
        case timestamp
    }
}
```

### 2. API Client (Point-Free Style)

```swift
import Foundation
import ComposableArchitecture

// MARK: - API Client Interface
struct InvoiceClassifierClient {
    var predictCategory: (InvoiceClassificationRequest) async throws -> CategoryPredictionResponse
    var predictTag: (InvoiceClassificationRequest) async throws -> TagPredictionResponse
    var health: () async throws -> HealthResponse
}

// MARK: - Live Implementation
extension InvoiceClassifierClient {
    private static let baseURL = "https://your-service.run.app"

    static let live = Self(
        predictCategory: { request in
            try await postRequest(request, endpoint: "/predict/category", responseType: CategoryPredictionResponse.self)
        },
        predictTag: { request in
            try await postRequest(request, endpoint: "/predict/tag", responseType: TagPredictionResponse.self)
        },
        health: {
            try await checkHealth()
        }
    )

    // MARK: - Network Implementation
    private static func postRequest<T: Decodable>(
        _ request: InvoiceClassificationRequest,
        endpoint: String,
        responseType: T.Type
    ) async throws -> T {
        let url = URL(string: "\(baseURL)\(endpoint)")!

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        urlRequest.timeoutInterval = 15.0 // Allow time for cold starts

        let (data, response) = try await URLSession.shared.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.init(rawValue: httpResponse.statusCode))
        }

        return try JSONDecoder().decode(T.self, from: data)
    }

    private static func checkHealth() async throws -> HealthResponse {
        let url = URL(string: "\(baseURL)/health")!

        var urlRequest = URLRequest(url: url)
        urlRequest.timeoutInterval = 10.0

        let (data, _) = try await URLSession.shared.data(for: urlRequest)
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }
}

// MARK: - Test/Mock Implementation
extension InvoiceClassifierClient {
    static let mock = Self(
        predictCategory: { request in
            CategoryPredictionResponse(
                probabilities: [
                    "marketing:ads": 0.85,
                    "operations:essential": 0.10,
                    "others:other": 0.05
                ],
                topCategory: "marketing:ads",
                topProbability: 0.85,
                modelVersion: "1.0.0"
            )
        },
        predictTag: { request in
            TagPredictionResponse(
                probabilities: [
                    "visual-panda": 0.75,
                    "referral-fee": 0.15,
                    "accounting": 0.10
                ],
                topTag: "visual-panda",
                topProbability: 0.75,
                modelVersion: "1.0.0"
            )
        },
        health: {
            HealthResponse(
                status: "healthy",
                modelLoaded: true,
                modelVersion: "1.0.0",
                timestamp: ISO8601DateFormatter().string(from: Date())
            )
        }
    )
}

// MARK: - Dependency Registration
extension DependencyValues {
    var invoiceClassifier: InvoiceClassifierClient {
        get { self[InvoiceClassifierClient.self] }
        set { self[InvoiceClassifierClient.self] = newValue }
    }
}

extension InvoiceClassifierClient: DependencyKey {
    static let liveValue = InvoiceClassifierClient.live
    static let testValue = InvoiceClassifierClient.mock
}
```

### 3. Usage Example

```swift
import ComposableArchitecture

@Reducer
struct InvoiceFeature {
    struct State: Equatable {
        var invoice: Invoice
        var predictedCategory: String?
        var predictedTag: String?
        var pendingRequests = 0
        var isLoading: Bool { pendingRequests > 0 }
    }

    enum Action {
        case classifyButtonTapped
        case categoryResponse(Result<CategoryPredictionResponse, Error>)
        case tagResponse(Result<TagPredictionResponse, Error>)
    }

    @Dependency(\.invoiceClassifier) var classifier

    var body: some ReducerOf<Self> {
        Reduce { state, action in
            switch action {
            case .classifyButtonTapped:
                state.pendingRequests = 2

                let request = InvoiceClassificationRequest(
                    entityId: state.invoice.entityId,
                    ownerId: state.invoice.ownerId,
                    netPrice: state.invoice.netPrice,
                    grossPrice: state.invoice.grossPrice,
                    currency: state.invoice.currency,
                    invoiceTitle: state.invoice.title, // Use full title, not just first word!
                    tin: state.invoice.tin,
                    issueDate: state.invoice.issueDate.formatted(.iso8601.year().month().day())
                )

                return .merge(
                    .run { send in
                        await send(.categoryResponse(
                            Result { try await classifier.predictCategory(request) }
                        ))
                    },
                    .run { send in
                        await send(.tagResponse(
                            Result { try await classifier.predictTag(request) }
                        ))
                    }
                )

            case let .categoryResponse(.success(response)):
                state.predictedCategory = response.topCategory
                state.pendingRequests -= 1
                return .none

            case let .tagResponse(.success(response)):
                state.predictedTag = response.topTag
                state.pendingRequests -= 1
                return .none

            case .categoryResponse(.failure):
                state.pendingRequests -= 1
                return .none

            case .tagResponse(.failure):
                state.pendingRequests -= 1
                return .none
            }
        }
    }
}
```

## Keep-Alive Scheduler Implementation

To avoid cold starts (9.5s), implement a background task that pings the `/health` endpoint every 4-5 minutes while your app is running.

### Point-Free Style Scheduler

```swift
import Foundation
import ComposableArchitecture

// MARK: - Keep-Alive Scheduler Client
struct KeepAliveScheduler {
    var start: () async -> Void
    var stop: () async -> Void
}

extension KeepAliveScheduler {
    static let live = Self(
        start: {
            await KeepAliveSchedulerLive.shared.start()
        },
        stop: {
            await KeepAliveSchedulerLive.shared.stop()
        }
    )
}

// MARK: - Live Implementation
actor KeepAliveSchedulerLive {
    static let shared = KeepAliveSchedulerLive()

    private var task: Task<Void, Never>?
    private let interval: TimeInterval = 240 // 4 minutes

    func start() {
        guard task == nil else { return }

        task = Task {
            while !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                    try await pingMLService()
                } catch {
                    print("Keep-alive ping failed: \(error)")
                }
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    private func pingMLService() async throws {
        let url = URL(string: "https://your-service.run.app/health")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 10.0

        _ = try await URLSession.shared.data(for: request)
    }
}

// MARK: - Dependency Registration
extension DependencyValues {
    var keepAliveScheduler: KeepAliveScheduler {
        get { self[KeepAliveScheduler.self] }
        set { self[KeepAliveScheduler.self] = newValue }
    }
}

extension KeepAliveScheduler: DependencyKey {
    static let liveValue = KeepAliveScheduler.live
    static let testValue = KeepAliveScheduler(
        start: {},
        stop: {}
    )
}
```

## Migration Checklist

### 1. Update Request Payload

All field names are now **snake_case**:

```swift
// ❌ OLD (camelCase JSON keys)
// entityId, ownerId, netPrice, grossPrice, issueDate

// ✅ NEW (snake_case JSON keys via CodingKeys)
// entity_id, owner_id, net_price, gross_price, issue_date
```

### 2. Update API Endpoints

```swift
// ❌ OLD (single endpoint)
// POST /predict

// ✅ NEW (dual endpoints)
// POST /predict/category — expense category prediction
// POST /predict/tag      — expense tag prediction
```

### 3. Update Invoice Title Handling

**Important:** The ML model uses the **full invoice title**, not just the first word!

```swift
// ❌ OLD (Modelbit - first word only)
let titleNormalized = invoice.title.components(separatedBy: " ").first ?? ""

// ✅ NEW (Cloud Run - full title)
let invoiceTitle = invoice.title // Use complete title
```

### 4. Implement Keep-Alive Scheduler

- [ ] Add KeepAliveScheduler client
- [ ] Start scheduler when app launches
- [ ] Stop scheduler when app terminates
- [ ] Test that scheduler pings every 4 minutes

### 5. Error Handling

```swift
// Handle cold starts (first request may take up to 15 seconds)
urlRequest.timeoutInterval = 15.0

// Handle failures gracefully
do {
    let categoryResponse = try await classifier.predictCategory(request)
    let tagResponse = try await classifier.predictTag(request)
} catch {
    print("Classification failed: \(error)")
}
```

### 6. Testing

- [ ] Test with various invoice types
- [ ] Verify full title is being sent (not normalized)
- [ ] Test timeout handling for cold starts
- [ ] Verify keep-alive scheduler is working
- [ ] Test both `/predict/category` and `/predict/tag` endpoints

## Example Categories (36)

Common categories:
- `marketing:ads` - Advertising and marketing campaigns
- `marketing:services` - Marketing agency services
- `operations:essential` - Essential software/tools (GitHub, Slack, etc.)
- `operations:ai` - AI/ML services (OpenAI, Anthropic, etc.)
- `operations:infrastructure` - Cloud infrastructure (AWS, GCP, etc.)
- `people:training` - Employee training and courses
- `people:benefits` - Employee benefits and perks
- `office:rent-and-administration` - Office rent
- `recruitment:services` - Recruitment agencies

Full list of 36 categories can be found in `models/category_model_metrics.json`.

## Example Tags (17)

- `accounting` - Accounting services
- `benefit-training` - Employee training benefits
- `benefit-multisport` - Multisport card benefits
- `benefit-medical-care` - Medical care benefits
- `benefit-english` - English lessons
- `benefit-books-formula` - Book budget
- `benefit-computer-formula` - Computer equipment budget
- `benefit-insurance` - Insurance benefits
- `benefit-outing` - Team outing benefits
- `benefit-psychologist` - Psychological support
- `benefit-apartments` - Apartment benefits
- `visual-panda` - Visual Panda design services
- `legal-advice` - Legal advisory services
- `referral-fee` - Employee referral fees
- `esop` - Employee stock option program
- `dashbit-jose-valim` - Dashbit consulting
- `BHP` - Occupational health & safety

Full list in `models/tag_model_metrics.json`.

## Performance Expectations

### Cold Start (15+ minutes of inactivity)
- **First request:** ~9.5 seconds
- **Breakdown:** Container init (3-4s) + Model load (4-5s) + Inference (0.2s)

### Warm Requests (with keep-alive scheduler)
- **All requests:** ~0.2 seconds

### Network Considerations
- Use 15 second timeout for requests (handles cold starts)
- Implement retry logic for production use
- Keep-alive scheduler uses silent failures (non-blocking)

## Health Check

```bash
curl https://your-service.run.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-01-07T12:00:00"
}
```

Note: `"healthy"` means **both** category and tag models are loaded.

## API Documentation

Interactive API docs: https://your-service.run.app/docs
