import time
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from ..providers.base import (
    BaseProvider,
    ProviderConfig,
    ProviderType,
    TaskType,
    LLMRequest,
    LLMResponse
)
from ..providers.factory import ProviderFactory

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    TASK_OPTIMIZED = "task_optimized"
    LEAST_LOADED = "least_loaded"

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: CircuitState = CircuitState.CLOSED
    half_open_calls: int = 0
    
    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False

@dataclass
class ProviderStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    total_cost_usd: float = 0
    last_used: Optional[datetime] = None
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_cost_per_request(self) -> float:
        if self.successful_requests == 0:
            return 0
        return self.total_cost_usd / self.successful_requests

@dataclass
class TaskRoutingConfig:
    preferred_providers: List[ProviderType] = field(default_factory=list)
    max_cost_per_request: float = 0.1
    max_latency_ms: float = 5000
    min_quality_score: float = 0.8

class Router:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[ProviderType, BaseProvider] = {}
        self.circuit_breakers: Dict[ProviderType, CircuitBreaker] = {}
        self.provider_stats: Dict[ProviderType, ProviderStats] = defaultdict(ProviderStats)
        
        self.strategy = RoutingStrategy(config.get('strategy', 'cost_optimized'))
        self.fallback_chain: List[ProviderType] = config.get('fallback_chain', [])
        self.task_routing: Dict[TaskType, TaskRoutingConfig] = {}
        
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        
        self._initialize_providers()
        self._initialize_task_routing()
    
    def _initialize_providers(self):
        providers_config = self.config.get('providers', [])
        
        for provider_config in providers_config:
            try:
                provider = ProviderFactory.create_dict(provider_config)
                provider_type = ProviderConfig(**provider_config).provider_type
                
                self.providers[provider_type] = provider
                self.circuit_breakers[provider_type] = CircuitBreaker(
                    failure_threshold=self.config.get('circuit_breaker', {}).get('failure_threshold', 5),
                    recovery_timeout=self.config.get('circuit_breaker', {}).get('recovery_timeout', 60)
                )
                
                logger.info(f"Initialized provider: {provider_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider: {e}")
    
    def _initialize_task_routing(self):
        task_routing_config = self.config.get('task_routing', {})
        
        for task_type_str, routing_config in task_routing_config.items():
            try:
                task_type = TaskType(task_type_str)
                self.task_routing[task_type] = TaskRoutingConfig(
                    preferred_providers=[ProviderType(p) for p in routing_config.get('preferred_providers', [])],
                    max_cost_per_request=routing_config.get('max_cost_per_request', 0.1),
                    max_latency_ms=routing_config.get('max_latency_ms', 5000),
                    min_quality_score=routing_config.get('min_quality_score', 0.8)
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid task routing config for {task_type_str}: {e}")
    
    def _get_available_providers(self) -> List[ProviderType]:
        available = []
        
        for provider_type, provider in self.providers.items():
            if not provider.config.enabled:
                continue
            
            circuit_breaker = self.circuit_breakers.get(provider_type)
            if circuit_breaker and not circuit_breaker.can_execute():
                continue
            
            available.append(provider_type)
        
        return available
    
    def _select_by_cost(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        if not providers:
            return None
        
        return min(
            providers,
            key=lambda p: self.providers[p].config.cost_per_1k_input
        )
    
    def _select_by_latency(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        if not providers:
            return None
        
        return min(
            providers,
            key=lambda p: self.provider_stats[p].avg_latency_ms or float('inf')
        )
    
    def _select_by_task(self, task_type: TaskType, providers: List[ProviderType]) -> Optional[ProviderType]:
        task_config = self.task_routing.get(task_type)
        
        if task_config and task_config.preferred_providers:
            for preferred in task_config.preferred_providers:
                if preferred in providers:
                    return preferred
        
        return providers[0] if providers else None
    
    def _select_round_robin(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        if not providers:
            return None
        
        selected = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return selected
    
    def _select_weighted(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        if not providers:
            return None
        
        weights = []
        for p in providers:
            stats = self.provider_stats[p]
            weight = 1.0 / (stats.avg_latency_ms + 1)
            weight *= (1 - stats.failure_rate) if hasattr(stats, 'failure_rate') else 1
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return providers[0]
        
        normalized_weights = [w / total_weight for w in weights]
        
        import random
        return random.choices(providers, weights=normalized_weights, k=1)[0]
    
    def _select_least_loaded(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        if not providers:
            return None
        
        return min(
            providers,
            key=lambda p: self.provider_stats[p].total_requests
        )
    
    async def select_provider(self, request: LLMRequest) -> Optional[ProviderType]:
        available = self._get_available_providers()
        
        if not available:
            logger.warning("No available providers")
            return None
        
        if self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._select_by_cost(available)
        elif self.strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return self._select_by_latency(available)
        elif self.strategy == RoutingStrategy.TASK_OPTIMIZED:
            return self._select_by_task(request.task_type, available)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)
        elif self.strategy == RoutingStrategy.WEIGHTED:
            return self._select_weighted(available)
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available)
        else:
            return available[0]
    
    async def route_request(self, request: LLMRequest) -> LLMResponse:
        selected_provider = await self.select_provider(request)
        
        if selected_provider is None:
            raise RuntimeError("No available providers for request")
        
        try:
            response = await self._execute_with_fallback(selected_provider, request)
            
            stats = self.provider_stats[selected_provider]
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_latency_ms += response.latency_ms
            stats.total_cost_usd += response.cost_usd
            stats.last_used = datetime.utcnow()
            
            circuit_breaker = self.circuit_breakers.get(selected_provider)
            if circuit_breaker:
                circuit_breaker.record_success()
            
            return response
            
        except Exception as e:
            stats = self.provider_stats[selected_provider]
            stats.total_requests += 1
            stats.failed_requests += 1
            
            circuit_breaker = self.circuit_breakers.get(selected_provider)
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            raise
    
    async def _execute_with_fallback(self, initial_provider: ProviderType, 
                                     request: LLMRequest) -> LLMResponse:
        providers_to_try = [initial_provider]
        
        for provider_type in self.fallback_chain:
            if provider_type != initial_provider and provider_type in self.providers:
                providers_to_try.append(provider_type)
        
        last_error = None
        
        for provider_type in providers_to_try:
            try:
                provider = self.providers[provider_type]
                response = await provider.generate(request)
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_type.value} failed: {e}")
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def route_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        selected_provider = await self.select_provider(request)
        
        if selected_provider is None:
            raise RuntimeError("No available providers for request")
        
        provider = self.providers[selected_provider]
        
        async for chunk in provider.generate_stream(request):
            yield chunk
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'providers': {
                provider_type.value: {
                    'enabled': provider.config.enabled,
                    'circuit_state': self.circuit_breakers[provider_type].state.value,
                    'stats': self.provider_stats[provider_type].__dict__
                }
                for provider_type, provider in self.providers.items()
            },
            'total_requests': sum(s.total_requests for s in self.provider_stats.values()),
            'total_cost': sum(s.total_cost_usd for s in self.provider_stats.values())
        }
    
    def reset_circuit_breaker(self, provider_type: ProviderType):
        if provider_type in self.circuit_breakers:
            self.circuit_breakers[provider_type] = CircuitBreaker(
                failure_threshold=self.circuit_breakers[provider_type].failure_threshold,
                recovery_timeout=self.circuit_breakers[provider_type].recovery_timeout
            )
            logger.info(f"Reset circuit breaker for {provider_type.value}")