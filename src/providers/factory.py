from typing import Dict, Any

from .base import BaseProvider, ProviderConfig, ProviderType
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .cohere_provider import CohereProvider

class ProviderFactory:
    _providers = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.COHERE: CohereProvider,
    }
    
    @classmethod
    def register_provider(cls, provider_type: ProviderType, provider_class: type):
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create(cls, config: ProviderConfig) -> BaseProvider:
        provider_class = cls._providers.get(config.provider_type)
        
        if provider_class is None:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")
        
        return provider_class(config)
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> BaseProvider:
        provider_type = ProviderType(config_dict['provider_type'])
        
        config = ProviderConfig(
            provider_type=provider_type,
            api_key=config_dict['api_key'],
            model=config_dict['model'],
            base_url=config_dict.get('base_url'),
            max_retries=config_dict.get('max_retries', 3),
            timeout=config_dict.get('timeout', 30.0),
            rate_limit_rpm=config_dict.get('rate_limit_rpm', 60),
            cost_per_1k_input=config_dict.get('cost_per_1k_input', 0.0),
            cost_per_1k_output=config_dict.get('cost_per_1k_output', 0.0),
            priority=config_dict.get('priority', 0),
            enabled=config_dict.get('enabled', True),
            metadata=config_dict.get('metadata', {})
        )
        
        return cls.create(config)