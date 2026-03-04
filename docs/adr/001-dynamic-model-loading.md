# ADR-001: Dynamic Multi-Model Loading via API

**Status:** Proposed
**Date:** 2026-03-04

## Context

Currently the classifier loads exactly two models at startup (category + tag) from a fixed path. All API requests use the same models.

As we onboard multiple clients or need per-entity models trained on different data, we need a way to serve multiple model versions from the same container without redeploying.

## Decision

Allow callers to specify a named model set via the API request. Models are organized in subdirectories and loaded on demand with an LRU cache.

### Directory structure

```
models/
  default/
    invoice_classifier.joblib
    invoice_tag_classifier.joblib
  client-abc/
    invoice_classifier.joblib
    invoice_tag_classifier.joblib
  experiment-v2/
    invoice_classifier.joblib
    invoice_tag_classifier.joblib
```

### API change

Add optional `model` field to `InvoiceRequest`, defaulting to `"default"`:

```json
{
  "model": "client-abc",
  "entity_id": "...",
  "net_price": 2500.0,
  ...
}
```

### Loading strategy

- Use `functools.lru_cache` (or similar) keyed by `(model_name, model_type)`.
- First request for a new model pays ~2s load cost; subsequent requests are instant.
- Set a max cache size (e.g. 10 model pairs) to cap memory usage.
- On startup, preload `default` model pair only.

### Benchmarks (measured on current models)

| Metric | Category model | Tag model |
|--------|---------------|-----------|
| Disk size | 12 MB | 6 MB |
| Cold load time | ~2,100 ms | ~13 ms |
| RAM per loaded model | ~30-50 MB | ~20-30 MB |
| Warm inference | <50 ms | <50 ms |

### Memory budget (1Gi container)

| Component | Estimate |
|-----------|----------|
| Python + dependencies | ~200 MB |
| Per model pair (loaded) | ~50-80 MB |
| Max model pairs in memory | ~10 |
| Request overhead | ~50 MB |

## Changes required

1. **`config.py`** — `MODEL_DIR` becomes the parent; model name selects subdirectory.
2. **`predict.py`** — Replace global `_cache` singletons with LRU cache keyed by model name.
3. **`main.py`** — Add optional `model: str = "default"` field to `InvoiceRequest`. Validate that the requested model directory exists (return 404 if not).
4. **Health endpoint** — Report which models are currently cached.
5. **Volume mount** — No change; the entire `models/` directory is already mounted.

## Consequences

**Positive:**
- Serve per-client or per-entity models without separate deployments.
- Enable A/B testing between model versions.
- Same container image, different model sets via mount.

**Negative:**
- First request for a new model has ~2s latency penalty.
- Memory usage grows with number of active models (mitigated by LRU eviction).
- Model directory must follow the naming convention.

**Risks:**
- If many distinct models are requested in quick succession, LRU eviction could cause frequent reloads. Mitigate by sizing the cache to expected active model count.

## Alternatives considered

1. **Separate container per model** — simpler isolation but higher infra cost and operational overhead.
2. **Download models from GCS on demand** — adds network latency on top of load time; more complex error handling.
3. **Preload all models at startup** — simplest but doesn't scale; startup time grows linearly with model count.
