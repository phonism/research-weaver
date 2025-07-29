# Research Weaver

An AI-powered research agent that performs iterative, in-depth research by intelligently searching, reading, and synthesizing information from multiple sources.

## Features

- 🔍 **Iterative Research**: Continuously refines search based on findings
- 📚 **Multi-source Integration**: Gathers information from various web sources
- 🤖 **LLM-powered Analysis**: Uses AI to understand and synthesize information
- 📊 **Visual Interface**: Streamlit-based UI for real-time research tracking
- 📝 **Structured Reports**: Generates comprehensive reports with citations

## Installation

### Prerequisites

- Python 3.8 or higher
- [DeepSeek API Key](https://platform.deepseek.com/) (for LLM)
- [Tavily API Key](https://tavily.com/) (for web search)

### Setup

```bash
# Clone the repository
git clone https://github.com/phonism/research-weaver.git
cd research-weaver

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys:
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

### Run the UI

```bash
streamlit run ui/app.py
```

### Programmatic Usage

```python
import asyncio
from src.supervisor import SupervisorAgent
from src.tools import create_llm_client, create_tools
from src.memory_store import MemoryStore

async def main():
    # Create components
    llm_client = create_llm_client()
    tools = create_tools()
    memory_store = MemoryStore()
    
    # Create supervisor
    supervisor = SupervisorAgent(
        llm_client=llm_client,
        model="deepseek-chat",
        tools=tools,
        memory_store=memory_store
    )
    
    # Conduct research
    result = await supervisor.research("Your research topic here")
    print(result)

# Run research
asyncio.run(main())
```

## How It Works

1. **Search**: Finds relevant sources based on the research query
2. **Read**: Extracts and parses content from selected sources
3. **Summarize**: Analyzes findings and identifies knowledge gaps
4. **Continue**: Iteratively searches for missing information
5. **Report**: Generates a comprehensive research report

## License

MIT License - see LICENSE file for details