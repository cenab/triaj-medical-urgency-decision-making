"""
Base provider class for LLM implementations.

This module provides the base class for all LLM provider implementations.
"""

import logging
import asyncio
from typing import Optional, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

class BaseProvider:
    """Base class for LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 150,
        api_base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base LLM provider.

        Args:
            api_key: API key for the model provider
            model: Model name/identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_base_url: Base URL for API (for alternative endpoints)
            **kwargs: Additional provider-specific arguments
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base_url = api_base_url
        self.kwargs = kwargs

    async def generate(self, prompt: str) -> str:
        """
        Generate text from the model (async implementation).

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")

    def generate_sync(self, prompt: str) -> str:
        """
        Generate text from the model (synchronous implementation).

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop is available, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a new loop if the current one is already running
            new_loop = asyncio.new_event_loop()
            result = new_loop.run_until_complete(self.generate(prompt))
            new_loop.close()
        else:
            result = loop.run_until_complete(self.generate(prompt))

        return result