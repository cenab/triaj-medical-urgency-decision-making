import os
# Remove unused imports: requests, json, aiohttp, asyncio
# Add google-generativeai imports
import google.generativeai as genai
from google.generativeai import types
from typing import List, Optional, Dict, Any
from .base import BaseProvider

import logging
# Setup logging
logger = logging.getLogger(__name__)

class GeminiProvider(BaseProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash", # Use a valid model name
        temperature: float = 0.7,
        max_tokens: int = 2048,
        # api_base_url: Optional[str] = None, # Removed
        system_prompt: Optional[str] = None,
        # top_p: float = 0.8, # Add if needed for GenerationConfig
        # top_k: int = 40,   # Add if needed for GenerationConfig
        **kwargs
    ):
        # Configure the SDK using the provided key or environment variable
        provided_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not provided_api_key:
            raise ValueError("API key must be provided either directly or through GOOGLE_API_KEY environment variable")
        genai.configure(api_key=provided_api_key)

        # Assuming BaseProvider does not strictly require api_key and model in __init__
        # or has been adapted. Remove super().__init__(api_key, model) if not needed.
        # super().__init__(api_key, model) # Original call

        self.model_name = model
        # Initialize the GenerativeModel client
        self.model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_prompt)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        logger.info(f"Initialized GeminiProvider with model: {self.model_name}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, system_prompt: {self.system_prompt}")
        # Store other generation config parameters if needed (e.g., top_p, top_k)

    # Remove old URL methods
    # def _get_model_url(self) -> str: ...
    # def _get_streaming_url(self) -> str: ...

    # Remove old payload preparation method
    # def _prepare_payload(self, prompt: str, system_instruction: Optional[str] = None) -> Dict[str, Any]: ...

    # Add helper methods for SDK configuration
    def _get_generation_config(self) -> types.GenerationConfig:
        config_args = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            # Add other parameters like top_p, top_k if needed
        }
        return types.GenerationConfig(**config_args)

    def _get_system_instruction_sdk_arg(self, system_instruction_override: Optional[str] = None) -> Optional[str]:
         """Helper to get the system instruction string for the SDK."""
         instruction_text = self.system_prompt
         return instruction_text

    async def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates content asynchronously using the google-generativeai SDK."""
        #sys_instruction_arg = self._get_system_instruction_sdk_arg(system_instruction)
        config = self._get_generation_config()
        logger.info(f"Generating content async with config: {config}")

        try:
            response = await self.model.generate_content_async(
                contents=[prompt], # Simple prompt needs to be in a list
                generation_config=config,
            )
            # Consider adding safety feedback handling: response.prompt_feedback
            if not response.parts:
                 # Handle cases where the response might be blocked or empty
                 logger.warning(f"Received empty response or blocked content. Feedback: {response.prompt_feedback}")
                 # Raise an exception or return a specific indicator based on feedback
                 raise Exception(f"No response from model, potentially blocked. Feedback: {response.prompt_feedback}")
            return response.text
        except Exception as e:
            logger.error(f"Error generating content asynchronously: {e}", exc_info=True)
            # Improve error handling based on specific SDK exceptions if needed
            raise Exception(f"Error generating response: {str(e)}")

    def generate_sync(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates content synchronously using the google-generativeai SDK."""
        #sys_instruction_arg = self._get_system_instruction_sdk_arg(system_instruction)
        config = self._get_generation_config()
        logger.info(f"Generating content sync with config: {config}")

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=config,
            )
            # Consider adding safety feedback handling: response.prompt_feedback
            if not response.parts:
                 logger.warning(f"Received empty response or blocked content. Feedback: {response.prompt_feedback}")
                 raise Exception(f"No response from model, potentially blocked. Feedback: {response.prompt_feedback}")
            return response.text
        except Exception as e:
            logger.error(f"Error generating content synchronously: {e}", exc_info=True)
            raise Exception(f"Error generating response: {str(e)}")

    def stream(self, prompt: str, system_instruction: Optional[str] = None):
        """Streams content using the google-generativeai SDK."""
        #sys_instruction_arg = self._get_system_instruction_sdk_arg(system_instruction)
        config = self._get_generation_config()
        logger.info(f"Streaming content with config: {config}")

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=config,
                stream=True
            )
            for chunk in response:
                 # Consider adding safety feedback handling: chunk.prompt_feedback
                 if not chunk.parts:
                     logger.warning(f"Received empty chunk or blocked content. Feedback: {chunk.prompt_feedback}")
                     # Decide whether to yield empty string, skip, or raise error
                     continue # Skip empty/blocked chunks for now
                 yield chunk.text
        except Exception as e:
            logger.error(f"Error streaming content: {e}", exc_info=True)
            raise Exception(f"Error streaming response: {str(e)}")

    # Helper function to convert chat message format
    def _convert_messages_to_sdk_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Converts a list of {'role': ..., 'content': ...} dicts to SDK's Content format."""
        sdk_contents = []
        # Construct the system message dictionary if a system prompt exists
        if self.system_prompt:
            # Note: Gemini API typically uses 'user' and 'model' roles in contents.
            # Prepending the system prompt as the first 'user' message here.
            # The role might need adjustment based on specific API behavior,
            # but 'user' aligns with starting a conversation context.
            # Consider if the first actual message should then be 'model'.
            sdk_contents.append({'role': 'system', 'parts': [{'text': self.system_prompt}]})

        for msg in messages:
            role = msg.get("role")
            # Assuming the text content is under the key 'content'
            content = msg.get("content")
            if not role or content is None: # Check for None explicitly
                logger.warning(f"Skipping message with missing role or content: {msg}")
                continue

            # Validate role - Gemini API expects 'user' and 'model'
            # Map 'assistant' to 'model' if necessary, otherwise enforce strict roles
            if role == "assistant":
                role = "model"
            elif role not in {"user", "model"}:
                 logger.error(f"Invalid role '{role}' in chat message. Must be 'user' or 'model'. Message: {msg}")
                 raise ValueError(f"Invalid role '{role}' in chat message. Must be 'user' or 'model'.")

            # Construct the dictionary directly
            sdk_contents.append({'role': role, 'parts': [{'text': content}]})
        return sdk_contents

    def chat(self, messages: List[Dict[str, str]], system_instruction: Optional[str] = None) -> str:
        """Generates chat response using the google-generativeai SDK, handling message history."""
        #sys_instruction_arg = self._get_system_instruction_sdk_arg(system_instruction)
        config = self._get_generation_config()

        try:
            # Convert the input message format to the SDK's expected format
            sdk_formatted_messages = self._convert_messages_to_sdk_format(messages)
            if not sdk_formatted_messages:
                raise ValueError("Input messages resulted in an empty list after conversion.")
        except ValueError as e:
            logger.error(f"Error processing chat messages for SDK: {e}", exc_info=True)
            raise e # Re-raise the validation error

        logger.info(f"Generating chat content with config: {config}")

        try:
            response = self.model.generate_content(
                contents=sdk_formatted_messages, # Use the converted message list
                generation_config=config,
            )
            # Consider adding safety feedback handling: response.prompt_feedback
            if not response.parts:
                 logger.warning(f"Received empty response or blocked content for chat. Feedback: {response.prompt_feedback}")
                 raise Exception(f"No response from model for chat, potentially blocked. Feedback: {response.prompt_feedback}")
            return response.text
        except Exception as e:
            logger.error(f"Error generating chat response: {e}", exc_info=True)
            raise Exception(f"Error generating chat response: {str(e)}")