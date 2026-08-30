import time
import json
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    AZURE = "azure"
    BEDROCK = "bedrock"
    LOCAL = "local"

class TaskType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE = "code"

@dataclass
class ProviderConfig:
    provider_type: ProviderType
    api_key: str
    model: str
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: float = 30.0
    rate_limit_rpm: int = 60
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMRequest:
    prompt: str
    task_type: TaskType = TaskType.CHAT
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt': self.prompt,
            'task_type': self.task_type.value,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'system_prompt': self.system_prompt,
            'stop_sequences': self.stop_sequences,
            'user_id': self.user_id,
            'request_id': self.request_id
        }
    
    def cache_key(self) -> str:
        key_parts = [
            self.prompt,
            self.task_type.value,
            str(self.max_tokens),
            str(self.temperature),
            self.system_prompt or ""
        ]
        return ":".join(key_parts)

@dataclass
class LLMResponse:
    text: str
    provider: ProviderType
    model: str
    usage: Dict[str, int]
    latency_ms: float
    cost_usd: float
    request_id: Optional[str] = None
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'provider': self.provider.value,
            'model': self.model,
            'usage': self.usage,
            'latency_ms': self.latency_ms,
            'cost_usd': self.cost_usd,
            'request_id': self.request_id,
            'cached': self.cached,
            'timestamp': self.timestamp
        }

class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.request_count = 0
        self.total_latency_ms = 0
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        pass
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost
    
    def record_success(self, latency_ms: float):
        self.request_count += 1
        self.total_latency_ms += latency_ms
    
    def record_error(self):
        self.error_count += 1
        self.last_error_time = datetime.utcnow()
    
    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0
        return self.total_latency_ms / self.request_count
    
    @property
    def error_rate(self) -> float:
        total = self.request_count + self.error_count
        if total == 0:
            return 0
        return self.error_count / total
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'provider': self.config.provider_type.value,
            'model': self.config.model,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'enabled': self.config.enabled
        }