# Research Weaver 🔬

An AI-powered multi-agent research system that conducts comprehensive, iterative research by intelligently searching, reading, and synthesizing information from multiple sources.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)

[简体中文](./README_CN.md) | English

## ✨ Features

### 🤖 Multi-Agent Architecture
- **Supervisor Agent**: Orchestrates research by creating specialized researcher agents
- **Researcher Agents**: Focused agents that handle specific research aspects
- **Dynamic Agent Creation**: Automatically spawns new researchers based on discovered topics

### 🔍 Intelligent Research Process
- **Iterative Search**: Continuously refines search queries based on findings
- **Multi-Source Integration**: Gathers information from various web sources
- **Knowledge Gap Detection**: Identifies missing information and creates targeted sub-researches
- **Citation Management**: Tracks and formats all sources with proper citations

### 💡 Advanced Capabilities
- **LLM-Powered Analysis**: Uses AI to understand context and synthesize information
- **Memory Management**: Persistent storage for research sessions with caching
- **Real-time Progress Tracking**: Visual interface shows live research progress
- **Structured Reports**: Generates comprehensive reports with clear sections and citations

### 🎯 Key Differentiators
- **Truly Autonomous**: Agents make independent decisions about what to research next
- **Context-Aware**: Each agent understands the overall research goal and its specific role
- **Scalable**: Can handle complex topics by breaking them into manageable sub-researches
- **Transparent**: Full visibility into the research process and decision-making

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [DeepSeek API Key](https://platform.deepseek.com/) (for LLM)
- [Tavily API Key](https://tavily.com/) or [Serper API Key](https://serper.dev/) (for web search)

### Installation

```bash
# Clone the repository
git clone https://github.com/phonism/research-weaver.git
cd research-weaver

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

Edit `.env` file with your API keys:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
# Optional: Use Serper as alternative search provider
# SERPER_API_KEY=your_serper_api_key_here
```

## 📖 Usage

### Web UI (Recommended)

```bash
streamlit run ui/app.py
```

Open your browser at `http://localhost:8501` to access the interactive research interface.

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

### Simple API Usage

```python
import asyncio
from src import research

# Using default settings (DeepSeek)
result = await research("Latest developments in quantum computing")

# Using OpenAI
result = await research(
    "Climate change solutions",
    provider="openai",
    api_key="your_openai_key",
    model="gpt-4-turbo-preview"
)
```

## 🔄 How It Works

### Research Flow

1. **Initial Planning**: Supervisor analyzes the research query and creates a research plan
2. **Agent Creation**: Specialized researcher agents are created for different aspects
3. **Iterative Research**: Each agent performs focused research through:
   - **Search**: Finding relevant sources using search tools
   - **Read**: Extracting and analyzing content from sources
   - **Synthesize**: Summarizing findings and identifying knowledge gaps
   - **Continue**: Creating new search queries for missing information
4. **Knowledge Integration**: Supervisor combines all findings into a coherent report
5. **Final Report**: Comprehensive document with all findings and citations

### Agent Communication

```
User Query → Supervisor Agent
                ↓
        Creates Multiple Researcher Agents
                ↓
    Each Researcher: Search → Read → Analyze → Report
                ↓
        Supervisor Integrates All Findings
                ↓
            Final Research Report
```

## 🏗️ Architecture

### Core Components

- **`src/supervisor.py`**: Main orchestrator that manages the research process
- **`src/researcher.py`**: Individual research agents with specialized focus
- **`src/tools.py`**: Search and read tools for information gathering
- **`src/memory_store.py`**: Persistent storage and caching system
- **`src/core.py`**: Base classes and data models
- **`ui/app.py`**: Streamlit-based web interface

### Technology Stack

- **LLM Integration**: OpenAI-compatible APIs (DeepSeek, OpenAI)
- **Search APIs**: Tavily, Serper
- **Web Scraping**: BeautifulSoup4, httpx
- **Async Processing**: asyncio for concurrent operations
- **UI Framework**: Streamlit for real-time visualization
- **Data Validation**: Pydantic for type safety

## 🎨 UI Features

### Real-time Visualization
- Live agent status tracking
- Progress bars for each research phase
- Expandable sections for detailed logs
- Citation tracking and management

### Interactive Controls
- Pause/Resume research capability
- Export results in multiple formats
- Search history and session management
- Customizable research parameters

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEEPSEEK_API_KEY` | DeepSeek API key for LLM | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes* |
| `SERPER_API_KEY` | Serper search API key | Yes* |
| `OPENAI_API_KEY` | OpenAI API key (optional) | No |

*Either Tavily or Serper API key is required

### Advanced Configuration

```python
# Custom LLM configuration
supervisor = SupervisorAgent(
    llm_client=llm_client,
    model="deepseek-chat",
    temperature=0.7,
    max_researchers=5,
    max_rounds=3
)

# Custom search parameters
tools = create_tools(
    search_max_results=10,
    read_timeout=30
)
```

## 📊 Example Research Topics

Research Weaver excels at complex, multi-faceted research topics:

- "Analyze the latest developments in renewable energy technology and market trends"
- "Compare different approaches to treating Alzheimer's disease in clinical trials"
- "Investigate the economic impact of AI on job markets across different industries"
- "Research the history and current state of quantum computing breakthroughs"

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Format code
black src/ ui/

# Lint code
ruff check src/ ui/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [DeepSeek](https://deepseek.com/) and [OpenAI](https://openai.com/) language models
- Search powered by [Tavily](https://tavily.com/) and [Serper](https://serper.dev/)
- UI framework by [Streamlit](https://streamlit.io/)

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/phonism/research-weaver/issues)
- **Discussions**: [GitHub Discussions](https://github.com/phonism/research-weaver/discussions)

---

Made with ❤️ by the Research Weaver Team