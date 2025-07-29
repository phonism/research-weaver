"""
LLM adapter for different providers (OpenAI, DeepSeek, etc.)
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters
    """
    
    @abstractmethod
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            messages: List of messages in OpenAI format
            model: Model name
            temperature: Temperature for generation
            
        Returns:
            Generated response text
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI API
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using OpenAI API
        """
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        return response.choices[0].message.content


class DeepSeekAdapter(LLMAdapter):
    """
    Adapter for DeepSeek API (public API)
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "deepseek-chat",
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using DeepSeek API
        """
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        return response.choices[0].message.content


def create_llm_adapter(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    **kwargs
) -> LLMAdapter:
    """
    Factory function to create LLM adapter
    
    Args:
        provider: LLM provider ("openai", "deepseek")
        api_key: API key for the provider
        **kwargs: Additional arguments
        
    Returns:
        LLM adapter instance
    """
    # Get API key from env if not provided
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIAdapter(api_key)
        
    elif provider == "deepseek":
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        return DeepSeekAdapter(api_key)
        
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Backward compatibility - create a client that works like AsyncOpenAI
class CompatibleLLMClient:
    """
    A client that mimics AsyncOpenAI interface but uses our adapter system
    """
    
    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter
        
    @property
    def chat(self):
        return self
        
    @property 
    def completions(self):
        return self
        
    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Create completion (mimics OpenAI interface)
        """
        content = await self.adapter.generate(messages, model, temperature)
        
        # Return object that mimics OpenAI response
        class MockChoice:
            def __init__(self, content):
                self.message = type('Message', (), {'content': content})()
                
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
                
        return MockResponse(content)