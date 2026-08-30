import time
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

from .base import BaseProvider, ProviderConfig, ProviderType, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class CohereProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(
                api_key=self.config.api_key
            )
        return self._client
    
    def _get_async_client(self):
        try:
            import cohere
            return cohere.AsyncClient(api_key=self.config.api_key)
        except ImportError:
            return self._get_client()
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        
        start_time = time.perf_counter()
        
        try:
            response = client.generate(
                model=self.config.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_sequences=request.stop_sequences or [],
                num_generations=1
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            text = response.generations[0].text.strip()
            
            prompt_tokens = self.count_tokens(request.prompt)
            completion_tokens = self.count_tokens(text)
            
            usage = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
            
            cost = self.calculate_cost(prompt_tokens, completion_tokens)
            
            self.record_success(latency_ms)
            
            return LLMResponse(
                text=text,
                provider=ProviderType.COHERE,
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
                request_id=request.request_id,
                metadata={
                    'id': response.id,
                    'prompt': getattr(response, 'prompt', None)
                }
            )
            
        except Exception as e:
            self.record_error()
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()
        
        response = client.generate(
            model=self.config.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_sequences=request.stop_sequences or [],
            stream=True
        )
        
        for event in response:
            if event.event_type == 'text-generation':
                yield event.text
    
    async def check_health(self) -> bool:
        try:
            client = self._get_client()
            response = client.generate(
                model=self.config.model,
                prompt="Hello",
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"Cohere health check failed: {e}")
            return False