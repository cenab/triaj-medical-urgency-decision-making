"""
Anthropic provider implementation.

This module provides API access to Anthropic Claude models.
"""

import os
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator, Union, Generator

from .base import BaseProvider

# Setup logging
logger = logging.getLogger(__name__)

class AnthropicProvider(BaseProvider):
    """Anthropic API provider implementation."""

    MAX_TOKENS = 100000  # Adjust based on the model's token limit (Claude has large context windows)

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model: Model name (default: claude-3-haiku-20240307)
            api_key: API key (optional, will use environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            api_base_url: Base URL for API (for non-standard endpoints)
            system_prompt: Optional system instructions for the model
            **kwargs: Additional Anthropic-specific arguments
        """
        super().__init__(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base_url=api_base_url,
            **kwargs
        )

        self.messages: List[Dict[str, str]] = []
        self.system_prompt = system_prompt
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic, AsyncAnthropic

            # Configure client settings
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.api_base_url:
                client_kwargs["base_url"] = self.api_base_url

            # Create clients
            self.client = Anthropic(**client_kwargs)
            self.async_client = AsyncAnthropic(**client_kwargs)

            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except ImportError:
            logger.error("Anthropic Python package not found. Install with: pip install anthropic")
            raise

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        # Don't add system messages to the messages list for Anthropic
        if role != "system":
            self.messages.append({"role": role, "content": content})
            self._ensure_token_limit()

    def _ensure_token_limit(self):
        """Ensure the conversation history does not exceed token limits."""
        try:
            # Use the token counting API to get accurate token counts
            count = self.client.beta.messages.count_tokens(
                model=self.model,
                messages=self.messages,
                system=self.system_prompt
            )
            total_tokens = count.input_tokens

            while total_tokens > self.MAX_TOKENS:
                # Remove the oldest user-assistant pair
                if len(self.messages) > 2:
                    self.messages.pop(0)
                    if len(self.messages) > 0 and self.messages[0]["role"] == "assistant":
                        self.messages.pop(0)
                else:
                    break

                # Recalculate token count
                count = self.client.beta.messages.count_tokens(
                    model=self.model,
                    messages=self.messages,
                    system=self.system_prompt
                )
                total_tokens = count.input_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}. Falling back to approximate counting.")
            # Fallback to approximate counting if the API call fails
            total_tokens = sum(len(m['content'].split()) for m in self.messages)
            if self.system_prompt:
                total_tokens += len(self.system_prompt.split())
            while total_tokens > self.MAX_TOKENS:
                if len(self.messages) > 2:
                    self.messages.pop(0)
                    if len(self.messages) > 0 and self.messages[0]["role"] == "assistant":
                        self.messages.pop(0)
                else:
                    break
                total_tokens = sum(len(m['content'].split()) for m in self.messages)
                if self.system_prompt:
                    total_tokens += len(self.system_prompt.split())

    async def generate(self, prompt: str, stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text from Anthropic model asynchronously.

        Args:
            prompt: Input prompt
            stream: Whether to stream the response

        Returns:
            Generated text or an async generator of text chunks if streaming
        """
        try:
            # Add user message to history
            self.add_message("user", prompt)

            # Prepare request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": self.messages,
                "stream": stream
            }

            # Add system prompt as a top-level parameter
            if self.system_prompt:
                params["system"] = self.system_prompt

            # Add any additional parameters from kwargs
            params.update({k: v for k, v in self.kwargs.items()
                         if k not in params})

            if stream:
                # Return a streaming response
                return self._stream_response(params)
            else:
                # Make the API call
                response = await self.async_client.messages.create(**params)

                # Extract the content
                assistant_message = response.content[0].text

                # Add assistant response to history
                self.add_message("assistant", assistant_message)

                return assistant_message

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise

    async def _stream_response(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle streaming responses from the Anthropic API."""
        try:
            async with self.async_client.messages.stream(**params) as stream:
                full_response = ""
                async for text in stream.text_stream:
                    full_response += text
                    yield text

                # Get the final message and add it to history
                message = await stream.get_final_message()
                assistant_message = message.content[0].text
                self.add_message("assistant", assistant_message)
        except Exception as e:
            logger.error(f"Error streaming from Anthropic API: {e}")
            raise

    def generate_sync(self, prompt: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from Anthropic model synchronously.

        Args:
            prompt: Input prompt
            stream: Whether to stream the response

        Returns:
            Generated text or a generator of text chunks if streaming
        """
        try:
            # Add user message to history
            self.add_message("user", prompt)

            # Prepare request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": self.messages,
                "stream": stream
            }

            # Add system prompt as a top-level parameter
            if self.system_prompt:
                params["system"] = self.system_prompt

            # Add any additional parameters from kwargs
            params.update({k: v for k, v in self.kwargs.items()
                         if k not in params})

            if stream:
                # Return a streaming response
                return self._stream_response_sync(params)
            else:
                # Make the API call
                response = self.client.messages.create(**params)

                # Extract the content
                assistant_message = response.content[0].text

                # Add assistant response to history
                self.add_message("assistant", assistant_message)

                return assistant_message
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise

    def _stream_response_sync(self, params: Dict[str, Any]) -> Generator[str, None, None]:
        """Handle streaming responses from the Anthropic API synchronously."""
        try:
            with self.client.messages.stream(**params) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield text

                # Get the final message and add it to history
                message = stream.get_final_message()
                assistant_message = message.content[0].text
                self.add_message("assistant", assistant_message)
        except Exception as e:
            logger.error(f"Error streaming from Anthropic API: {e}")
            raise