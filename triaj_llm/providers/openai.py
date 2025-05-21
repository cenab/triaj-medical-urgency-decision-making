"""
OpenAI API provider implementation.

This module provides API access to OpenAI models.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from .base import BaseProvider

# Setup logging
logger = logging.getLogger(__name__)

class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation."""

    MAX_TOKENS = 4096  # Adjust based on the model's token limit

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 150,
        api_base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            api_base_url: Base URL for API (for non-standard endpoints).
            system_prompt: Optional system instructions for the model.
            **kwargs: Additional OpenAI-specific arguments.
        """
        super().__init__(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base_url=api_base_url,
            system_prompt=system_prompt,
            **kwargs
        )
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI, AsyncOpenAI

            # Configure base client settings
            client_kwargs = {
                "api_key": self.api_key,
                "base_url": self.api_base_url
            }

            # Create clients
            self.client = OpenAI(**client_kwargs)
            self.async_client = AsyncOpenAI(**client_kwargs)

            logger.info(f"Initialized OpenAI client with model: {self.model}")
        except ImportError:
            logger.error("OpenAI Python package not found. Install with: pip install openai")
            raise

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        self._ensure_token_limit()

    def _ensure_token_limit(self):
        """Ensure the conversation history does not exceed token limits."""
        total_tokens = sum(len(m['content'].split()) for m in self.messages)
        while total_tokens > self.MAX_TOKENS:
            # Remove the oldest user-assistant pair
            if len(self.messages) > 2:
                self.messages.pop(1)
                self.messages.pop(1)
            else:
                break
            total_tokens = sum(len(m['content'].split()) for m in self.messages)

    async def generate(self, prompt: str) -> str:
        """
        Generate text from OpenAI model asynchronously.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.
        """
        try:
            self.add_message("user", prompt)
            # Prepare request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": self.messages
            }

            # Add any additional parameters from kwargs
            params.update({k: v for k, v in self.kwargs.items() if k not in params})

            # Make the API call
            response = await self.async_client.chat.completions.create(**params)

            # Extract and return the content
            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)
            return assistant_message

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    def generate_sync(self, prompt: str) -> str:
        """
        Generate text from OpenAI model synchronously.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.
        """
        try:
            self.add_message("user", prompt)
            # Prepare request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": self.messages
            }

            # Add any additional parameters from kwargs
            params.update({k: v for k, v in self.kwargs.items() if k not in params})

            # Make the API call
            response = self.client.chat.completions.create(**params)

            # Extract and return the content
            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)
            return assistant_message

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise