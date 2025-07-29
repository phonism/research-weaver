"""
Global memory store for sharing data between agents and UI
"""
from typing import Any, Dict, List
from datetime import datetime
import threading


class MemoryStore:
    """
    Thread-safe global memory store for sharing data across agents and UI
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def set(self, key: str, value: Any):
        """Set a value in memory"""
        with self._lock:
            self._data[key] = value
            
    def get(self, key: str, default=None):
        """Get a value from memory"""
        with self._lock:
            return self._data.get(key, default)
            
    def append_to_list(self, key: str, value: Any):
        """Append to a list in memory (create if not exists)"""
        with self._lock:
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(value)
            
    def get_list(self, key: str) -> List[Any]:
        """Get a list from memory"""
        with self._lock:
            return self._data.get(key, []).copy()
            
    def clear(self):
        """Clear all memory"""
        with self._lock:
            self._data.clear()
            
    def add_llm_response(self, agent_role: str, response: str, model: str = ""):
        """Convenience method to add LLM response"""
        response_data = {
            "agent_role": agent_role,
            "response": response,
            "model": model,
            "timestamp": datetime.now()
        }
        self.append_to_list("llm_responses", response_data)
        
    def get_llm_responses(self) -> List[Dict]:
        """Get all LLM responses"""
        return self.get_list("llm_responses")
        
    def add_search_results(self, agent_role: str, query: str, results: List[Dict]):
        """Add search results to memory"""
        search_data = {
            "agent_role": agent_role,
            "query": query,
            "results": results,
            "timestamp": datetime.now()
        }
        self.append_to_list("search_results", search_data)
        
    def get_search_results(self) -> List[Dict]:
        """Get all search results"""
        return self.get_list("search_results")
        
    def add_read_operation(self, agent_role: str, url: str, status: str, summary: str = "", original_content: str = ""):
        """Add read operation status to memory"""
        read_data = {
            "agent_role": agent_role,
            "url": url,
            "status": status,  # "reading", "summarizing", "completed"
            "summary": summary,
            "original_content": original_content,
            "timestamp": datetime.now()
        }
        self.append_to_list("read_operations", read_data)
        
    def get_read_operations(self) -> List[Dict]:
        """Get all read operations"""
        return self.get_list("read_operations")


# Global instance
memory_store = MemoryStore()