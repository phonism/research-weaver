"""
Core base classes for the LLM-driven research agent system
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from .llm_adapter import LLMAdapter, CompatibleLLMClient
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """
    Role of a message in conversation
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    A single message in the conversation
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfoPiece:
    """
    A piece of information collected during research
    """
    content: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMInteraction:
    """
    Record of an LLM interaction (request and response)
    """
    agent_role: str
    messages: List[Dict[str, str]]
    response: str
    model: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentState(str, Enum):
    """
    Current state of an agent
    """
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(str, Enum):
    """
    Types of actions an agent can take
    """
    USE_TOOL = "use_tool"
    CREATE_AGENT = "create_agent"
    CONTINUE_THINKING = "continue_thinking"
    RETURN_RESULT = "return_result"


@dataclass
class AgentAction:
    """
    An action to be taken by the agent
    """
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """
    Response from an agent's step
    """
    thought: str
    action: Optional[AgentAction] = None
    result: Optional[str] = None
    needs_specialist: bool = False
    specialist_role: Optional[str] = None
    sub_task: Optional[str] = None
    is_complete: bool = False


class Context:
    """
    Agent's working context including conversation history and collected information
    """
    def __init__(self, initial_goal: str = ""):
        self.conversation_history: List[Message] = []
        self.collected_info: List[InfoPiece] = []
        self.llm_interactions: List[LLMInteraction] = []  # Store LLM interactions
        self.current_goal: str = initial_goal
        self.working_memory: Dict[str, Any] = {}
        self.created_at: datetime = datetime.now()
        
    def add_message(self, role: MessageRole, content: str, **metadata):
        """
        Add a message to conversation history
        """
        self.conversation_history.append(
            Message(role=role, content=content, metadata=metadata)
        )
        
    def add_info(self, info: Union[InfoPiece, List[InfoPiece]]):
        """
        Add collected information
        """
        if isinstance(info, list):
            self.collected_info.extend(info)
        else:
            self.collected_info.append(info)
            
    def add_llm_interaction(self, interaction: LLMInteraction):
        """
        Add LLM interaction to history
        """
        self.llm_interactions.append(interaction)
            
    def get_recent_messages(self, n: int = 30) -> List[Message]:
        """
        Get n most recent messages, prioritizing important ones
        """
        # For researcher agents, be more selective about message history
        if hasattr(self, 'current_goal') and len(self.conversation_history) > n:
            # Always include the initial goal
            important_messages = []
            recent_messages = []
            
            for msg in self.conversation_history:
                # Keep system messages about research results and goals
                if (msg.role == MessageRole.USER or 
                    "summarized" in msg.content or 
                    "Search results" in msg.content or
                    "【文章总结】" in msg.content or
                    "has completed their research" in msg.content or
                    "Full findings:" in msg.content or
                    "参考文献" in msg.content):
                    important_messages.append(msg)
                else:
                    recent_messages.append(msg)
            
            # Combine important messages + recent ones, limit total
            selected = important_messages + recent_messages[-(n-len(important_messages)):]
            return selected[-n:]
        
        return self.conversation_history[-n:]
        
    def get_context_summary(self) -> str:
        """
        Get a summary of current context for LLM
        """
        summary = f"Current Goal: {self.current_goal}\n\n"
        
        if self.collected_info:
            summary += "Research Progress Summary:\n"
            
            # Group by source type
            searches = [info for info in self.collected_info if "Search results" in info.content]
            reads = [info for info in self.collected_info if "Content from" in info.content or "summarized" in info.metadata.get("type", "")]
            
            if searches:
                summary += f"- Completed {len(searches)} searches on topics: "
                search_topics = []
                for search in searches[-3:]:  # Last 3 searches
                    if "Search results for" in search.content:
                        topic = search.content.split("'")[1] if "'" in search.content else "unknown"
                        search_topics.append(topic[:30])
                summary += ", ".join(search_topics) + "\n"
            
            if reads:
                summary += f"- Read and analyzed {len(reads)} sources including: "
                sources = [info.source for info in reads[-3:]]  # Last 3 sources
                summary += ", ".join(sources) + "\n"
            
            summary += f"- Total information pieces collected: {len(self.collected_info)}\n"
        
        summary += f"\nWorking Memory: {self.working_memory}\n"
        
        return summary


class LLMAgent(ABC):
    """
    Base class for all LLM-driven agents
    """
    def __init__(
        self, 
        role: str,
        system_prompt: str,
        llm_client: Optional[CompatibleLLMClient] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory_store=None
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.memory_store = memory_store
        self.context = Context()
        self.state = AgentState.IDLE
        self.available_tools = {}
        self.child_agents = []
        
    async def step(self) -> AgentResponse:
        """
        Execute one step of agent reasoning
        """
        self.state = AgentState.THINKING
        
        # Build prompt with current context
        prompt = self._build_prompt()
        
        # Get LLM response
        llm_output = await self._query_llm(prompt)
        
        # Store LLM interaction in memory store if available
        if self.memory_store:
            self.memory_store.add_llm_response(
                agent_role=self.role,
                response=llm_output,
                model=self.model
            )
        
        # Parse LLM output to determine action
        response = self._parse_llm_output(llm_output)
        
        # Update conversation history with LLM output first
        self.context.add_message(MessageRole.ASSISTANT, llm_output)
        
        # Execute action if needed
        if response.action:
            self.state = AgentState.ACTING
            await self._execute_action(response.action)
        
        # Update state
        if response.is_complete:
            self.state = AgentState.COMPLETED
        else:
            self.state = AgentState.IDLE
            
        return response
        
    async def run_until_complete(self, max_steps: int = 50) -> str:
        """
        Run the agent until it completes its task
        """
        steps = 0
        final_result = ""
        
        while self.state != AgentState.COMPLETED and steps < max_steps:
            response = await self.step()
            
            if response.result:
                final_result = response.result
                
            steps += 1
            
        return final_result
        
    def _build_prompt(self) -> str:
        """
        Build the prompt for LLM including system prompt and context
        """
        prompt = f"{self.system_prompt}\n\n"
        context_summary = self.context.get_context_summary()
        prompt += f"Context:\n{context_summary}\n\n"
        
        # Add recent conversation history
        recent_messages = self.context.get_recent_messages()
        
        import logging
        import hashlib
        
        # Generate hashes for comparison
        context_hash = hashlib.md5(context_summary.encode()).hexdigest()
        
        logging.info(f"[BUILD_PROMPT] Agent: {self.role}")
        logging.info(f"[BUILD_PROMPT] Context Summary Hash: {context_hash}")
        logging.info(f"[BUILD_PROMPT] Context Summary Length: {len(context_summary)}")
        logging.info(f"[BUILD_PROMPT] Recent Messages Count: {len(recent_messages)}")
        logging.info(f"[BUILD_PROMPT] Total Conversation History: {len(self.context.conversation_history)} messages")
        
        # Log first few characters of context for comparison
        logging.info(f"[BUILD_PROMPT] Context Summary (first 200 chars): {context_summary[:200]}...")
        
        prompt += "Recent Conversation:\n"
        for i, msg in enumerate(recent_messages):
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            logging.debug(f"[{self.role}] Message {i}: {msg.role.value} - {content_preview}")
            # For important messages, log more details
            if i < 3:  # Log first 3 messages in detail
                logging.info(f"[BUILD_PROMPT] Message {i}: Role={msg.role.value}, Length={len(msg.content)}, Preview={content_preview}")
            prompt += f"{msg.role.value}: {msg.content}\n"
            
        prompt += "\nWhat should I do next? Think step by step."
        
        # Final prompt hash for comparison
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        logging.info(f"[BUILD_PROMPT] Final Prompt Hash: {prompt_hash}")
        logging.info(f"[BUILD_PROMPT] Final Prompt Length: {len(prompt)}")
        
        return prompt
        
    async def _query_llm(self, prompt: str) -> str:
        """
        Query the LLM with the given prompt
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized")
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        import logging
        import hashlib
        import json
        
        # Generate a hash for comparison across runs
        messages_hash = hashlib.md5(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        
        logging.info(f"[LLM_CALL] Agent: {self.role}")
        logging.info(f"[LLM_CALL] Model: {self.model}, Temperature: {self.temperature}")
        logging.info(f"[LLM_CALL] Messages Hash: {messages_hash}")
        logging.info(f"[LLM_CALL] System Prompt (first 100 chars): {self.system_prompt[:100]}...")
        logging.info(f"[LLM_CALL] User Prompt (first 200 chars): {prompt[:200]}...")
        logging.info(f"[LLM_CALL] User Prompt Length: {len(prompt)}")
        
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        response_content = response.choices[0].message.content
        
        logging.info(f"[LLM_RESPONSE] Agent: {self.role}, Response Length: {len(response_content)}")
        
        return response_content
        
    @abstractmethod
    def _parse_llm_output(self, output: str) -> AgentResponse:
        """
        Parse LLM output to extract thought and action
        Must be implemented by subclasses
        """
        pass
        
    @abstractmethod  
    async def _execute_action(self, action: AgentAction):
        """
        Execute the given action
        Must be implemented by subclasses
        """
        pass
        
    def register_tool(self, name: str, tool_func):
        """
        Register a tool that the agent can use
        """
        self.available_tools[name] = tool_func
        
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names
        """
        return list(self.available_tools.keys())