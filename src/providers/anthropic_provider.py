import time
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

from .base import BaseProvider, ProviderConfig, ProviderType, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        return self._client
    
    def _get_async_client(self):
        if self._async_client is None:
            import anthropic
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        return self._async_client
    
    def count_tokens(self, text: str) -> int:
        try:
            client = self._get_client()
            return client.count_tokens(text).input_tokens
        except Exception:
            return len(text) // 4
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._get_async_client()
        
        messages = [{"role": "user", "content": request.prompt}]
        
        kwargs = {
            'model': self.config.model,
            'max_tokens': request.max_tokens,
            'messages': messages
        }
        
        if request.system_prompt:
            kwargs['system'] = request.system_prompt
        
        if request.temperature is not None:
            kwargs['temperature'] = request.temperature
        
        if request.stop_sequences:
            kwargs['stop_sequences'] = request.stop_sequences
        
        start_time = time.perf_counter()
        
        try:
            response = await client.messages.create(**kwargs)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            text = response.content[0].text if response.content else ""
            
            usage = {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            
            cost = self.calculate_cost(
                usage['prompt_tokens'],
                usage['completion_tokens']
            )
            
            self.record_success(latency_ms)
            
            return LLMResponse(
                text=text,
                provider=ProviderType.ANTHROPIC,
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
                request_id=request.request_id,
                metadata={
                    'stop_reason': response.stop_reason,
                    'id': response.id
                }
            )
            
        except Exception as e:
            self.record_error()
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_async_client()
        
        messages = [{"role": "user", "content": request.prompt}]
        
        kwargs = {
            'model': self.config.model,
            'max_tokens': request.max_tokens,
            'messages': messages
        }
        
        if request.system_prompt:
            kwargs['system'] = request.system_prompt
        
        if request.temperature is not None:
            kwargs['temperature'] = request.temperature
        
        if request.stop_sequences:
            kwargs['stop_sequences'] = request.stop_sequences
        
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def check_health(self) -> bool:
        try:
            client = self._get_async_client()
            response = await client.messages.create(
                model=self.config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False