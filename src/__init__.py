# Research Weaver - AI-powered research agent
import os
from typing import Optional
from openai import AsyncOpenAI

from .supervisor import SupervisorAgent
from .researcher import ResearcherAgent
from .tools import SearchTool, ReadTool, create_tools
from .core import Context, InfoPiece


async def research(
    query: str, 
    provider: str = "openai",
    api_key: Optional[str] = None, 
    model: str = "gpt-4-turbo-preview"
) -> str:
    """
    Conduct AI-powered research on a given topic
    
    Args:
        query: Research question or topic
        provider: LLM provider ("openai", "deepseek")
        api_key: API key (uses env var if not provided)
        model: LLM model to use
        
    Returns:
        Comprehensive research report
    """
    from .llm_adapter import create_llm_adapter, CompatibleLLMClient
    
    # Create LLM adapter
    adapter = create_llm_adapter(provider=provider, api_key=api_key)
    llm_client = CompatibleLLMClient(adapter)
    
    # Create supervisor with tools
    supervisor = SupervisorAgent(
        llm_client=llm_client,
        model=model
    )
    
    # Conduct research
    result = await supervisor.research(query)
    
    return result


__all__ = [
    "research",
    "SupervisorAgent", 
    "ResearcherAgent",
    "SearchTool",
    "ReadTool",
    "Context",
    "InfoPiece"
]