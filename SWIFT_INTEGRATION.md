# Swift Integration Guide: Migrating from Modelbit to Cloud Run

This guide helps you refactor your Swift application to use the payroll invoice classifier deployed on Google Cloud Run instead of Modelbit.

## Overview

**Old Service:** Modelbit (offline)
**New Service:** Google Cloud Run
**Service URL:** `https://your-service.run.app`
**Framework:** FastAPI (compatible with any HTTP client)
**Response Format:** JSON

## API Endpoint

### Predict Invoice Category

**Endpoint:** `POST /predict`
**URL:** `https://your-service.run.app/predict`
**Content-Type:** `application/json`
**Authentication:** None (public endpoint)

### Request Payload

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

### Response Format

```json
{
  "probabilities": {
    "marketing:ads": 0.9998,
    "marketing:services": 0.0001,
    "operations:essential": 0.0001,
    "...": "..."
  },
  "top_category": "marketing:ads",
  "top_probability": 0.9998,
  "model_version": "1.0.0"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entityId` | String | Yes | Unique identifier for the company/entity |
| `ownerId` | String | Yes | Unique identifier for the invoice owner |
| `netPrice` | Double | Yes | Net price excluding VAT (must be > 0) |
| `grossPrice` | Double | Yes | Gross price including VAT (must be > 0) |
| `currency` | String | Yes | 3-letter currency code (PLN, USD, EUR, GBP) |
| `invoice_title` | String | Yes | Full invoice title/description (not just first word) |
| `tin` | String? | No | Tax identification number (can be null or empty string) |
| `issueDate` | String | Yes | Invoice issue date in YYYY-MM-DD format |

**Response Fields:**
- `probabilities`: Dictionary of all category names with their probability scores (0.0 to 1.0)
- `top_category`: The category with the highest probability
- `top_probability`: The probability score of the top category
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
        case entityId
        case ownerId
        case netPrice
        case grossPrice
        case currency
        case invoiceTitle = "invoice_title"
        case tin
        case issueDate
    }
}

// MARK: - Response Model
struct InvoiceClassificationResponse: Codable {
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
    var predict: (InvoiceClassificationRequest) async throws -> InvoiceClassificationResponse
    var health: () async throws -> HealthResponse
}

// MARK: - Live Implementation
extension InvoiceClassifierClient {
    static let live = Self(
        predict: { request in
            try await classifyInvoice(request)
        },
        health: {
            try await checkHealth()
        }
    )

    // MARK: - Network Implementation
    private static func classifyInvoice(_ request: InvoiceClassificationRequest) async throws -> InvoiceClassificationResponse {
        let url = URL(string: "https://your-service.run.app/predict")!

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

        return try JSONDecoder().decode(InvoiceClassificationResponse.self, from: data)
    }

    private static func checkHealth() async throws -> HealthResponse {
        let url = URL(string: "https://your-service.run.app/health")!

        var urlRequest = URLRequest(url: url)
        urlRequest.timeoutInterval = 10.0

        let (data, _) = try await URLSession.shared.data(for: urlRequest)
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }
}

// MARK: - Test/Mock Implementation
extension InvoiceClassifierClient {
    static let mock = Self(
        predict: { request in
            // Return mock response for testing
            InvoiceClassificationResponse(
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
        var isLoading = false
    }

    enum Action {
        case classifyButtonTapped
        case classificationResponse(Result<InvoiceClassificationResponse, Error>)
    }

    @Dependency(\.invoiceClassifier) var classifier

    var body: some ReducerOf<Self> {
        Reduce { state, action in
            switch action {
            case .classifyButtonTapped:
                state.isLoading = true

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

                return .run { send in
                    await send(.classificationResponse(
                        Result { try await classifier.predict(request) }
                    ))
                }

            case let .classificationResponse(.success(response)):
                state.isLoading = false
                state.predictedCategory = response.topCategory
                return .none

            case .classificationResponse(.failure):
                state.isLoading = false
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
import Combine
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
                    // Silent failures - don't crash if ping fails
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

### Integrate Scheduler in App Lifecycle

```swift
import ComposableArchitecture

@Reducer
struct AppFeature {
    struct State: Equatable {
        // ... your app state
    }

    enum Action {
        case onAppear
        case onDisappear
    }

    @Dependency(\.keepAliveScheduler) var scheduler

    var body: some ReducerOf<Self> {
        Reduce { state, action in
            switch action {
            case .onAppear:
                return .run { _ in
                    await scheduler.start()
                }

            case .onDisappear:
                return .run { _ in
                    await scheduler.stop()
                }
            }
        }
    }
}
```

### Alternative: Combine-Based Timer

If you prefer Combine over async/await:

```swift
import Combine
import Foundation

final class KeepAliveSchedulerCombine {
    private var cancellable: AnyCancellable?
    private let url = URL(string: "https://your-service.run.app/health")!

    func start() {
        cancellable = Timer.publish(every: 240, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.ping()
            }
    }

    func stop() {
        cancellable?.cancel()
        cancellable = nil
    }

    private func ping() {
        var request = URLRequest(url: url)
        request.timeoutInterval = 10.0

        URLSession.shared.dataTask(with: request) { _, _, error in
            if let error = error {
                print("Keep-alive ping failed: \(error)")
            }
        }.resume()
    }
}
```

## Migration Checklist

### 1. Replace Modelbit Code

- [ ] Remove Modelbit SDK/dependencies
- [ ] Remove Modelbit API keys from configuration
- [ ] Add new API client code (InvoiceClassifierClient)
- [ ] Update request payload to match new format

### 2. Update Invoice Title Handling

**Important:** The ML model uses the **full invoice title**, not just the first word!

```swift
// ❌ OLD (Modelbit - first word only)
let titleNormalized = invoice.title.components(separatedBy: " ").first ?? ""

// ✅ NEW (Cloud Run - full title)
let invoiceTitle = invoice.title // Use complete title
```

### 3. Update Date Formatting

```swift
// Ensure date is in YYYY-MM-DD format
let dateFormatter = ISO8601DateFormatter()
dateFormatter.formatOptions = [.withFullDate]
let issueDate = dateFormatter.string(from: invoice.date)
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
    let response = try await classifier.predict(request)
    // Success
} catch {
    // Fallback: retry once or show user message
    print("Classification failed: \(error)")
}
```

### 6. Testing

- [ ] Test with various invoice types
- [ ] Verify full title is being sent (not normalized)
- [ ] Test timeout handling for cold starts
- [ ] Verify keep-alive scheduler is working
- [ ] Test with network disconnected (error handling)

## Performance Expectations

### Cold Start (15+ minutes of inactivity)
- **First request:** ~9.5 seconds
- **Breakdown:** Container init (3-4s) + Model load (4-5s) + Inference (0.2s)

### Warm Requests (with keep-alive scheduler)
- **All requests:** ~0.2 seconds
- **Consistency:** Scheduler pings every 4 min → service stays warm

### Network Considerations
- Use 15 second timeout for requests (handles cold starts)
- Implement retry logic for production use
- Keep-alive scheduler uses silent failures (non-blocking)

## Example Categories

The model predicts 36 different expense categories:

**Common Categories:**
- `marketing:ads` - Advertising and marketing campaigns
- `marketing:services` - Marketing agency services
- `operations:essential` - Essential software/tools (GitHub, Slack, etc.)
- `operations:ai` - AI/ML services (OpenAI, Anthropic, etc.)
- `operations:infrastructure` - Cloud infrastructure (AWS, GCP, etc.)
- `people:training` - Employee training and courses
- `people:benefits` - Employee benefits and perks
- `office:rent-and-administration` - Office rent
- `recruitment:services` - Recruitment agencies

Full list of 36 categories can be found in `models/model_metrics.json`.

## Support & Troubleshooting

### Common Issues

**1. Timeout Errors**
- Increase timeout to 15s for first request after inactivity
- Implement keep-alive scheduler to avoid cold starts

**2. Invalid Response Format**
- Ensure you're sending `invoice_title` (full title), not `title_normalized`
- Check date format is YYYY-MM-DD
- Verify all required fields are present

**3. Low Confidence Scores**
- Model returns all 36 category probabilities
- Use `top_category` and `top_probability` for decision making
- Consider threshold (e.g., only accept if `top_probability > 0.5`)

### Health Check

Test the service is running:

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

### API Documentation

Interactive API docs available at:
https://your-service.run.app/docs

## Cost & Scaling

- **Current usage:** $0.00/month (within free tier)
- **Free tier limits:** 2M requests/month, 180k vCPU-seconds/month
- **Your usage:** ~20-50 requests/day = ~1,500/month (0.075% of limit)
- **Keep-alive cost:** $0.00 (HTTP requests within free tier)

**No action needed** - service will scale automatically within free tier limits.

## Questions?

For issues with the ML service itself, check the main repository README.md and troubleshooting sections.

For Swift integration questions, share this file with your development team or AI assistant.
