"""
Factory module for creating provider instances.

This module provides a factory function to create provider instances
based on the provider name.
"""

import logging
from typing import Optional, Dict, Any

from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .deepseek import DeepSeekProvider
from .gemini import GeminiProvider

# Setup logging
logger = logging.getLogger(__name__)

def create_provider(
    provider_name: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 150,
    api_base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    Create an LLM provider based on name.

    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'deepseek', 'gemini')
        model: Model name specific to the provider
        api_key: API key for the provider
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_base_url: Base URL for API (for non-standard endpoints)
        system_prompt: Optional system instructions for the model
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured provider instance
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4",
            temperature=temperature,
            max_tokens=max_tokens,
            api_base_url=api_base_url,
            system_prompt=system_prompt,
            **kwargs
        )
    elif provider_name in ["anthropic", "claude"]:
        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-3-5-sonnet-latest",
            temperature=temperature,
            max_tokens=max_tokens,
            api_base_url=api_base_url,
            system_prompt=system_prompt,
            **kwargs
        )
    elif provider_name in ["deepseek", "deepseek-ai"]:
        return DeepSeekProvider(
            api_key=api_key,
            model=model or "deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens,
            api_base_url=api_base_url,
            system_prompt=system_prompt,
            **kwargs
        )
    elif provider_name in ["google", "gemini"]:
        return GeminiProvider(
            api_key=api_key,
            model=model or "gemini-1.5-pro-latest",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")