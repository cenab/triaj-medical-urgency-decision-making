"""
DeepSeek provider implementation.

This module provides API access to DeepSeek models.
"""

import os
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
import json

from .base import BaseProvider

# Setup logging
logger = logging.getLogger(__name__)

class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider implementation."""

    MAX_TOKENS = 32000  # Adjust based on the model's token limit

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 150,
        api_base_url: Optional[str] = "https://api.deepseek.com",
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model: Model name (e.g., 'deepseek-chat', 'deepseek-reasoner')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_base_url: Base URL for API (default: https://api.deepseek.com)
            system_prompt: Optional system instructions for the model
            stream: Whether to stream the response
            **kwargs: Additional DeepSeek-specific arguments
        """
        super().__init__(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base_url=api_base_url,
            **kwargs
        )

        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        self.stream = stream
        self._init_client()

    def _init_client(self):
        """Initialize DeepSeek client."""
        try:
            from openai import OpenAI, AsyncOpenAI

            # Configure base URL and headers
            self.api_base_url = self.api_base_url or "https://api.deepseek.com"

            if not self.api_key:
                raise ValueError("DeepSeek API key is required")

            # Create clients for sync and async requests
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )

            logger.info(f"Initialized DeepSeek client with model: {self.model}")
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
            if len(self.messages) > 2 and self.messages[0]["role"] != "system":
                self.messages.pop(0)
                if len(self.messages) > 0 and self.messages[0]["role"] == "assistant":
                    self.messages.pop(0)
            elif len(self.messages) > 2:  # If the first message is a system message
                self.messages.pop(1)
                if len(self.messages) > 1:
                    self.messages.pop(1)
            else:
                break
            total_tokens = sum(len(m['content'].split()) for m in self.messages)

    async def generate(self, prompt: str) -> str:
        """
        Generate text from DeepSeek model asynchronously.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            # Add user message to history
            self.add_message("user", prompt)

            # Make the API call
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=self.stream,
                **self.kwargs
            )

            # Extract the content
            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self.add_message("assistant", assistant_message)

            return assistant_message

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            raise

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate text from DeepSeek model asynchronously with streaming.

        Args:
            prompt: Input prompt

        Yields:
            Generated text chunks
        """
        try:
            # Add user message to history
            self.add_message("user", prompt)

            # Make the API call
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **self.kwargs
            )

            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            # Add the complete response to history
            self.add_message("assistant", full_response)

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            raise

    def generate_sync(self, prompt: str) -> str:
        """
        Generate text from DeepSeek model synchronously.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            # Add user message to history
            self.add_message("user", prompt)

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=self.stream,
                **self.kwargs
            )

            # Extract the content
            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self.add_message("assistant", assistant_message)

            return assistant_message

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            raise