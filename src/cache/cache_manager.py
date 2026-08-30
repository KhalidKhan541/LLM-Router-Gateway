import json
import hashlib
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict
import time

from ..providers.base import LLMResponse, ProviderType

logger = logging.getLogger(__name__)

class CacheEntry:
    def __init__(self, response: LLMResponse, ttl_seconds: int = 3600):
        self.response = response
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.last_accessed = time.time()
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()

class LRUCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[LLMResponse]:
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                return None
            
            entry.access()
            self._cache.move_to_end(key)
            
            return entry.response
    
    async def set(self, key: str, response: LLMResponse, ttl_seconds: int = 3600):
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(response, ttl_seconds)
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self):
        async with self._lock:
            self._cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for e in self._cache.values() if e.is_expired)
            total_access = sum(e.access_count for e in self._cache.values())
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'total_access_count': total_access,
                'max_size': self.max_size
            }

class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 prefix: str = "llm_cache:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client
    
    async def get(self, key: str) -> Optional[LLMResponse]:
        try:
            client = self._get_client()
            data = await client.get(f"{self.prefix}{key}")
            
            if data is None:
                return None
            
            response_data = json.loads(data)
            
            return LLMResponse(
                text=response_data['text'],
                provider=ProviderType(response_data['provider']),
                model=response_data['model'],
                usage=response_data['usage'],
                latency_ms=response_data['latency_ms'],
                cost_usd=response_data['cost_usd'],
                request_id=response_data.get('request_id'),
                cached=True,
                metadata=response_data.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, response: LLMResponse, ttl_seconds: int = 3600):
        try:
            client = self._get_client()
            
            response_data = {
                'text': response.text,
                'provider': response.provider.value,
                'model': response.model,
                'usage': response.usage,
                'latency_ms': response.latency_ms,
                'cost_usd': response.cost_usd,
                'request_id': response.request_id,
                'metadata': response.metadata,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            await client.setex(
                f"{self.prefix}{key}",
                ttl_seconds,
                json.dumps(response_data)
            )
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        try:
            client = self._get_client()
            result = await client.delete(f"{self.prefix}{key}")
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self):
        try:
            client = self._get_client()
            keys = await client.keys(f"{self.prefix}*")
            if keys:
                await client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            info = await client.info('stats')
            keyspace = await client.info('keyspace')
            
            return {
                'connected': True,
                'total_keys': info.get('keyspace', {}).get('db0', {}).get('keys', 0),
                'hits': info.get('stats', {}).get('keyspace_hits', 0),
                'misses': info.get('stats', {}).get('keyspace_misses', 0)
            }
        except Exception as e:
            return {'connected': False, 'error': str(e)}

class CacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_type = config.get('type', 'memory')
        self.default_ttl = config.get('default_ttl', 3600)
        self.enabled = config.get('enabled', True)
        
        if self.cache_type == 'redis':
            self.cache = RedisCache(
                redis_url=config.get('redis_url', 'redis://localhost:6379'),
                prefix=config.get('prefix', 'llm_cache:')
            )
        else:
            max_size = config.get('max_size', 1000)
            self.cache = LRUCache(max_size=max_size)
    
    def _generate_cache_key(self, request_dict: Dict[str, Any]) -> str:
        key_parts = [
            request_dict.get('prompt', ''),
            request_dict.get('task_type', ''),
            str(request_dict.get('max_tokens', '')),
            str(request_dict.get('temperature', ''))
        ]
        
        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get(self, request_dict: Dict[str, Any]) -> Optional[LLMResponse]:
        if not self.enabled:
            return None
        
        cache_key = self._generate_cache_key(request_dict)
        return await self.cache.get(cache_key)
    
    async def set(self, request_dict: Dict[str, Any], response: LLMResponse, 
                  ttl_seconds: Optional[int] = None):
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(request_dict)
        ttl = ttl_seconds or self.default_ttl
        await self.cache.set(cache_key, response, ttl)
    
    async def invalidate(self, request_dict: Dict[str, Any]) -> bool:
        cache_key = self._generate_cache_key(request_dict)
        return await self.cache.delete(cache_key)
    
    async def clear(self):
        await self.cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        stats = await self.cache.get_stats()
        stats['type'] = self.cache_type
        stats['enabled'] = self.enabled
        stats['default_ttl'] = self.default_ttl
        return stats