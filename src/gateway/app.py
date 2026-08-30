import time
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..providers.base import LLMRequest, LLMResponse, TaskType, ProviderType
from ..routing.router import Router, RoutingStrategy
from ..cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    prompt: str
    task_type: str = "chat"
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    user_id: Optional[str] = None
    use_cache: bool = True
    stream: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "task_type": "chat",
                "max_tokens": 512,
                "temperature": 0.7
            }
        }

class GenerationResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    cost_usd: float
    request_id: Optional[str] = None
    cached: bool = False
    timestamp: str

class BatchRequest(BaseModel):
    requests: List[GenerationRequest]
    parallel: bool = True

class BatchResponse(BaseModel):
    responses: List[GenerationResponse]
    total_latency_ms: float
    total_cost_usd: float

class RouterStats(BaseModel):
    strategy: str
    providers: Dict[str, Any]
    total_requests: int
    total_cost: float
    cache_stats: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    providers: Dict[str, bool]
    timestamp: str

class LLMGateway:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.router = Router(config.get('routing', {}))
        self.cache = CacheManager(config.get('cache', {}))
        self.app = FastAPI(
            title="LLM Router Gateway",
            description="Multi-provider LLM routing with cost/latency optimization",
            version="1.0.0"
        )
        
        self._setup_routes()
        self._setup_middleware()
        
        self.request_count = 0
        self.total_latency_ms = 0
        self.total_cost_usd = 0
    
    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        @self.app.post("/v1/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
            request_id = str(uuid.uuid4())
            
            task_type = TaskType(request.task_type)
            
            llm_request = LLMRequest(
                prompt=request.prompt,
                task_type=task_type,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                stop_sequences=request.stop_sequences,
                user_id=request.user_id,
                request_id=request_id
            )
            
            if request.use_cache:
                cached_response = await self.cache.get(llm_request.to_dict())
                if cached_response:
                    return GenerationResponse(
                        text=cached_response.text,
                        provider=cached_response.provider.value,
                        model=cached_response.model,
                        usage=cached_response.usage,
                        latency_ms=cached_response.latency_ms,
                        cost_usd=cached_response.cost_usd,
                        request_id=request_id,
                        cached=True,
                        timestamp=cached_response.timestamp
                    )
            
            start_time = time.perf_counter()
            
            try:
                response = await self.router.route_request(llm_request)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                self.request_count += 1
                self.total_latency_ms += latency_ms
                self.total_cost_usd += response.cost_usd
                
                if request.use_cache:
                    background_tasks.add_task(
                        self.cache.set,
                        llm_request.to_dict(),
                        response
                    )
                
                return GenerationResponse(
                    text=response.text,
                    provider=response.provider.value,
                    model=response.model,
                    usage=response.usage,
                    latency_ms=latency_ms,
                    cost_usd=response.cost_usd,
                    request_id=request_id,
                    cached=False,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/generate/stream")
        async def generate_stream(request: GenerationRequest):
            request_id = str(uuid.uuid4())
            
            task_type = TaskType(request.task_type)
            
            llm_request = LLMRequest(
                prompt=request.prompt,
                task_type=task_type,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                stop_sequences=request.stop_sequences,
                user_id=request.user_id,
                request_id=request_id
            )
            
            async def event_generator():
                try:
                    async for chunk in self.router.route_stream(llm_request):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: error: {str(e)}\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )
        
        @self.app.post("/v1/generate/batch", response_model=BatchResponse)
        async def generate_batch(batch: BatchRequest):
            start_time = time.perf_counter()
            
            if batch.parallel:
                tasks = [self._generate_single(req) for req in batch.requests]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                responses = []
                for req in batch.requests:
                    try:
                        response = await self._generate_single(req)
                        responses.append(response)
                    except Exception as e:
                        responses.append(GenerationResponse(
                            text=f"Error: {str(e)}",
                            provider="error",
                            model="error",
                            usage={},
                            latency_ms=0,
                            cost_usd=0,
                            cached=False,
                            timestamp=datetime.utcnow().isoformat()
                        ))
            
            total_latency = (time.perf_counter() - start_time) * 1000
            total_cost = sum(r.cost_usd for r in responses)
            
            return BatchResponse(
                responses=responses,
                total_latency_ms=total_latency,
                total_cost_usd=total_cost
            )
        
        @self.app.get("/v1/stats", response_model=RouterStats)
        async def get_stats():
            router_stats = self.router.get_stats()
            cache_stats = await self.cache.get_stats()
            
            return RouterStats(
                strategy=router_stats['strategy'],
                providers=router_stats['providers'],
                total_requests=router_stats['total_requests'],
                total_cost=router_stats['total_cost'],
                cache_stats=cache_stats
            )
        
        @self.app.get("/v1/health", response_model=HealthResponse)
        async def health_check():
            providers_health = {}
            
            for provider_type, provider in self.router.providers.items():
                try:
                    health = await provider.check_health()
                    providers_health[provider_type.value] = health
                except Exception:
                    providers_health[provider_type.value] = False
            
            all_healthy = all(providers_health.values()) if providers_health else False
            
            return HealthResponse(
                status="healthy" if all_healthy else "degraded",
                providers=providers_health,
                timestamp=datetime.utcnow().isoformat()
            )
        
        @self.app.post("/v1/cache/invalidate")
        async def invalidate_cache(request: GenerationRequest):
            llm_request = LLMRequest(
                prompt=request.prompt,
                task_type=TaskType(request.task_type),
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            invalidated = await self.cache.invalidate(llm_request.to_dict())
            
            return {"invalidated": invalidated}
        
        @self.app.post("/v1/cache/clear")
        async def clear_cache():
            await self.cache.clear()
            return {"status": "cache cleared"}
        
        @self.app.post("/v1/providers/{provider_type}/reset")
        async def reset_provider(provider_type: str):
            try:
                pt = ProviderType(provider_type)
                self.router.reset_circuit_breaker(pt)
                return {"status": f"reset {provider_type}"}
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid provider type")
    
    async def _generate_single(self, request: GenerationRequest) -> GenerationResponse:
        request_id = str(uuid.uuid4())
        
        task_type = TaskType(request.task_type)
        
        llm_request = LLMRequest(
            prompt=request.prompt,
            task_type=task_type,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            stop_sequences=request.stop_sequences,
            user_id=request.user_id,
            request_id=request_id
        )
        
        start_time = time.perf_counter()
        
        response = await self.router.route_request(llm_request)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationResponse(
            text=response.text,
            provider=response.provider.value,
            model=response.model,
            usage=response.usage,
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
            request_id=request_id,
            cached=False,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def get_app(self):
        return self.app