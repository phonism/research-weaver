"""
Research Weaver Usage Example
Demonstrates how to use the research system programmatically
"""
import asyncio
import os
from src.supervisor import SupervisorAgent
from src.tools import create_llm_client, create_tools
from src.memory_store import MemoryStore


async def main():
    """
    Example of using Research Weaver programmatically
    """
    # Ensure API keys are set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: Please set DEEPSEEK_API_KEY environment variable")
        print("Get your API key from: https://platform.deepseek.com/")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: Please set TAVILY_API_KEY environment variable")
        print("Get your API key from: https://tavily.com/")
        return
    
    print("🔬 Research Weaver Example")
    print("=" * 50)
    
    try:
        # Create LLM client
        print("Creating LLM client...")
        llm_client = create_llm_client()
        
        # Create tools
        print("Creating search and read tools...")
        tools = create_tools()
        
        # Create memory store
        memory_store = MemoryStore()
        
        # Create supervisor
        print("Creating supervisor agent...")
        supervisor = SupervisorAgent(
            llm_client=llm_client,
            model="deepseek-chat",
            tools=tools,
            memory_store=memory_store
        )
        
        # Research query
        query = "Analyze the latest developments in renewable energy technology and market trends"
        
        print(f"\\nStarting research on: {query}")
        print("=" * 50)
        
        # Conduct research
        result = await supervisor.research(query)
        
        print("\\n📄 Research Complete!")
        print("=" * 50)
        print(result)
        
        # Save result to file
        with open("research_result.md", "w", encoding="utf-8") as f:
            f.write(result)
        
        print("\\n✅ Result saved to research_result.md")
        
        # Get research summary
        summary = supervisor.get_research_summary()
        print(f"\\n📊 Research Summary:")
        print(f"Total researchers created: {summary['total_researchers']}")
        print(f"Tasks completed: {len(summary['tasks'])}")
        print(f"Information pieces collected: {summary['collected_info']}")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Research Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())