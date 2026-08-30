import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.providers.base import (
    ProviderConfig,
    ProviderType,
    TaskType,
    LLMRequest,
    LLMResponse,
    BaseProvider
)
from src.routing.router import (
    Router,
    RoutingStrategy,
    CircuitBreaker,
    CircuitState,
    ProviderStats
)
from src.cache.cache_manager import CacheManager, LRUCache

@pytest.fixture
def sample_config():
    return {
        'routing': {
            'strategy': 'cost_optimized',
            'fallback_chain': ['openai', 'anthropic'],
            'circuit_breaker': {
                'failure_threshold': 3,
                'recovery_timeout': 30
            },
            'providers': [
                {
                    'provider_type': 'openai',
                    'api_key': 'test-key',
                    'model': 'gpt-4',
                    'cost_per_1k_input': 0.03,
                    'cost_per_1k_output': 0.06,
                    'enabled': True
                }
            ]
        },
        'cache': {
            'type': 'memory',
            'enabled': True,
            'default_ttl': 300,
            'max_size': 100
        }
    }

@pytest.fixture
def sample_request():
    return LLMRequest(
        prompt="Test prompt",
        task_type=TaskType.CHAT,
        max_tokens=100,
        temperature=0.7
    )

@pytest.fixture
def sample_response():
    return LLMResponse(
        text="Test response",
        provider=ProviderType.OPENAI,
        model="gpt-4",
        usage={'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30},
        latency_ms=150.0,
        cost_usd=0.001
    )

class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_record_success(self):
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_record_failure_opens_circuit(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
    
    def test_can_execute_closed(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True
    
    def test_can_execute_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        
        assert cb.can_execute() is False
    
    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.utcnow()
        
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

class TestProviderStats:
    def test_avg_latency(self):
        stats = ProviderStats()
        stats.total_requests = 10
        stats.successful_requests = 10
        stats.total_latency_ms = 1000
        
        assert stats.avg_latency_ms == 100
    
    def test_success_rate(self):
        stats = ProviderStats()
        stats.total_requests = 10
        stats.successful_requests = 8
        
        assert stats.success_rate == 0.8
    
    def test_avg_cost(self):
        stats = ProviderStats()
        stats.successful_requests = 10
        stats.total_cost_usd = 0.5
        
        assert stats.avg_cost_per_request == 0.05

class TestLRUCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = LRUCache(max_size=10)
        response = LLMResponse(
            text="test",
            provider=ProviderType.OPENAI,
            model="gpt-4",
            usage={},
            latency_ms=100,
            cost_usd=0.001
        )
        
        await cache.set("key1", response)
        result = await cache.get("key1")
        
        assert result is not None
        assert result.text == "test"
    
    @pytest.mark.asyncio
    async def test_expired_entry(self):
        cache = LRUCache(max_size=10)
        response = LLMResponse(
            text="test",
            provider=ProviderType.OPENAI,
            model="gpt-4",
            usage={},
            latency_ms=100,
            cost_usd=0.001
        )
        
        await cache.set("key1", response, ttl_seconds=0)
        result = await cache.get("key1")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        cache = LRUCache(max_size=2)
        response = LLMResponse(
            text="test",
            provider=ProviderType.OPENAI,
            model="gpt-4",
            usage={},
            latency_ms=100,
            cost_usd=0.001
        )
        
        await cache.set("key1", response)
        await cache.set("key2", response)
        await cache.set("key3", response)
        
        result = await cache.get("key1")
        assert result is None
        
        result = await cache.get("key3")
        assert result is not None

class TestCacheManager:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        config = {'type': 'memory', 'enabled': True, 'default_ttl': 300, 'max_size': 100}
        cache = CacheManager(config)
        
        response = LLMResponse(
            text="test",
            provider=ProviderType.OPENAI,
            model="gpt-4",
            usage={},
            latency_ms=100,
            cost_usd=0.001
        )
        
        request_dict = {'prompt': 'test', 'task_type': 'chat'}
        
        await cache.set(request_dict, response)
        result = await cache.get(request_dict)
        
        assert result is not None
        assert result.cached is True

class TestLLMRequest:
    def test_to_dict(self):
        request = LLMRequest(
            prompt="Test prompt",
            task_type=TaskType.CHAT,
            max_tokens=100
        )
        
        result = request.to_dict()
        
        assert result['prompt'] == "Test prompt"
        assert result['task_type'] == 'chat'
        assert result['max_tokens'] == 100
    
    def test_cache_key(self):
        request = LLMRequest(
            prompt="Test prompt",
            task_type=TaskType.CHAT,
            max_tokens=100,
            temperature=0.7
        )
        
        key = request.cache_key()
        
        assert isinstance(key, str)
        assert "Test prompt" in key

class TestLLMResponse:
    def test_to_dict(self):
        response = LLMResponse(
            text="Test response",
            provider=ProviderType.OPENAI,
            model="gpt-4",
            usage={'prompt_tokens': 10},
            latency_ms=150.0,
            cost_usd=0.001
        )
        
        result = response.to_dict()
        
        assert result['text'] == "Test response"
        assert result['provider'] == 'openai'
        assert result['latency_ms'] == 150.0