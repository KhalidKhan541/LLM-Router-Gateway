# LLM Router Gateway

A production-grade multi-provider LLM routing gateway with cost/latency optimization, fallback chains, circuit breakers, and response caching.

## Overview

This gateway routes LLM requests across multiple providers (OpenAI, Anthropic, Cohere) based on:
- **Cost**: Minimize API expenses
- **Latency**: Optimize response times
- **Task Type**: Route to best provider for task
- **Load**: Distribute requests evenly
- **Reliability**: Automatic fallback on failures

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Router Gateway                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Request    │───▶│   Routing    │───▶│    Cache     │      │
│  │   Handler    │    │   Engine     │    │    Layer     │      │
│  └──────────────┘    └──────┬───────┘    └──────────────┘      │
│                             │                                   │
│                    ┌────────┴────────┐                          │
│                    │                 │                          │
│                    ▼                 ▼                          │
│           ┌──────────────┐  ┌──────────────┐                   │
│           │   Circuit    │  │   Provider   │                   │
│           │   Breaker    │  │   Manager    │                   │
│           └──────────────┘  └──────┬───────┘                   │
│                                    │                            │
│                    ┌───────────────┼───────────────┐           │
│                    │               │               │           │
│                    ▼               ▼               ▼           │
│           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│           │    OpenAI    │ │  Anthropic   │ │    Cohere    │   │
│           └──────────────┘ └──────────────┘ └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Cohere (extensible)
- **Smart Routing**: Cost, latency, task-type, round-robin, weighted
- **Fallback Chains**: Automatic failover to secondary providers
- **Circuit Breakers**: Prevent cascade failures
- **Response Caching**: LRU (memory) or Redis
- **Streaming Support**: SSE for real-time responses
- **Batch Processing**: Parallel request execution
- **Health Monitoring**: Provider health checks
- **Cost Tracking**: Per-request and aggregate costs

## Quick Start

### Installation

```bash
git clone https://github.com/KhalidKhan541/LLM-Router-Gateway.git
cd LLM-Router-Gateway
pip install -r requirements.txt
```

### Configuration

1. Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export COHERE_API_KEY="your-cohere-key"
```

2. Edit `configs/gateway_config.yaml`:

```yaml
routing:
  strategy: "cost_optimized"
  fallback_chain:
    - "openai"
    - "anthropic"

providers:
  - provider_type: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    cost_per_1k_input: 0.03
```

### Run Gateway

```bash
# Development
python -m src.gateway.app

# Production
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Generate Text

```bash
POST /v1/generate
```

```json
{
  "prompt": "Explain quantum computing",
  "task_type": "chat",
  "max_tokens": 512,
  "temperature": 0.7
}
```

### Stream Response

```bash
POST /v1/generate/stream
```

Returns Server-Sent Events (SSE).

### Batch Generation

```bash
POST /v1/generate/batch
```

```json
{
  "requests": [
    {"prompt": "Request 1", "max_tokens": 100},
    {"prompt": "Request 2", "max_tokens": 100}
  ],
  "parallel": true
}
```

### Get Statistics

```bash
GET /v1/stats
```

### Health Check

```bash
GET /v1/health
```

### Clear Cache

```bash
POST /v1/cache/clear
```

## Routing Strategies

### Cost Optimized

Routes to cheapest provider while meeting quality requirements.

```yaml
routing:
  strategy: "cost_optimized"
```

### Latency Optimized

Routes to fastest provider based on historical latency.

```yaml
routing:
  strategy: "latency_optimized"
```

### Task Optimized

Routes based on task type (chat, code, embedding).

```yaml
routing:
  strategy: "task_optimized"
  task_routing:
    code:
      preferred_providers: ["openai"]
```

### Round Robin

Distributes requests evenly across providers.

```yaml
routing:
  strategy: "round_robin"
```

### Weighted

Routes based on provider weights (latency, success rate).

```yaml
routing:
  strategy: "weighted"
```

### Least Loaded

Routes to provider with fewest active requests.

```yaml
routing:
  strategy: "least_loaded"
```

## Fallback Configuration

```yaml
routing:
  fallback_chain:
    - "openai"      # Primary
    - "anthropic"   # Secondary
    - "cohere"      # Tertiary
```

If primary fails, automatically tries next provider.

## Circuit Breaker

Prevents cascade failures by temporarily disabling failing providers.

```yaml
routing:
  circuit_breaker:
    failure_threshold: 5    # Failures before opening
    recovery_timeout: 60    # Seconds before retry
```

States:
- **Closed**: Normal operation
- **Open**: Provider disabled
- **Half-Open**: Testing recovery

## Caching

### Memory Cache (LRU)

```yaml
cache:
  type: "memory"
  max_size: 10000
  default_ttl: 3600
```

### Redis Cache

```yaml
cache:
  type: "redis"
  redis_url: "redis://localhost:6379"
  default_ttl: 3600
```

## Provider Configuration

```yaml
providers:
  - provider_type: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    cost_per_1k_input: 0.03
    cost_per_1k_output: 0.06
    rate_limit_rpm: 60
    priority: 1
    enabled: true
```

## Usage Examples

### Python Client

```python
import httpx

response = httpx.post("http://localhost:8000/v1/generate", json={
    "prompt": "What is machine learning?",
    "task_type": "chat",
    "max_tokens": 256
})

print(response.json()["text"])
```

### cURL

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain neural networks",
    "max_tokens": 512
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem", "stream": true}'
```

## Monitoring

### Statistics Endpoint

```bash
curl http://localhost:8000/v1/stats
```

Response:

```json
{
  "strategy": "cost_optimized",
  "providers": {
    "openai": {
      "enabled": true,
      "circuit_state": "closed",
      "stats": {
        "total_requests": 150,
        "successful_requests": 145,
        "avg_latency_ms": 450,
        "avg_cost_per_request": 0.003
      }
    }
  },
  "total_requests": 150,
  "total_cost": 0.45
}
```

### Health Check

```bash
curl http://localhost:8000/v1/health
```

## Project Structure

```
llm-router-gateway/
├── src/
│   ├── providers/
│   │   ├── base.py              # Provider abstraction
│   │   ├── factory.py           # Provider factory
│   │   ├── openai_provider.py   # OpenAI adapter
│   │   ├── anthropic_provider.py# Anthropic adapter
│   │   └── cohere_provider.py   # Cohere adapter
│   ├── routing/
│   │   └── router.py            # Routing engine
│   ├── cache/
│   │   └── cache_manager.py     # Caching layer
│   └── gateway/
│       └── app.py               # FastAPI gateway
├── configs/
│   └── gateway_config.yaml      # Configuration
├── tests/
│   └── test_router.py           # Unit tests
├── requirements.txt
└── README.md
```

## License

MIT License