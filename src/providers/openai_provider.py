import time
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

from .base import BaseProvider, ProviderConfig, ProviderType, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self._client
    
    def _get_async_client(self):
        if self._async_client is None:
            import openai
            self._async_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self._async_client
    
    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model)
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._get_async_client()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        start_time = time.perf_counter()
        
        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop_sequences
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            choice = response.choices[0]
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            cost = self.calculate_cost(
                usage['prompt_tokens'],
                usage['completion_tokens']
            )
            
            self.record_success(latency_ms)
            
            return LLMResponse(
                text=choice.message.content,
                provider=ProviderType.OPENAI,
                model=self.config.model,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
                request_id=request.request_id,
                metadata={
                    'finish_reason': choice.finish_reason,
                    'system_fingerprint': getattr(response, 'system_fingerprint', None)
                }
            )
            
        except Exception as e:
            self.record_error()
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_async_client()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop_sequences,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def check_health(self) -> bool:
        try:
            client = self._get_async_client()
            response = await client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False