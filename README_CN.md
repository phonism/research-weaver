# Research Weaver 🔬

一个由AI驱动的多智能体研究系统，通过智能搜索、阅读和综合多源信息进行全面、迭代的研究。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)

简体中文 | [English](./README.md)

## ✨ 特性

### 🤖 多智能体架构
- **监督者智能体**：通过创建专门的研究员智能体来协调研究
- **研究员智能体**：处理特定研究方面的专注型智能体
- **动态智能体创建**：根据发现的主题自动生成新的研究员

### 🔍 智能研究流程
- **迭代搜索**：基于发现不断优化搜索查询
- **多源整合**：从各种网络来源收集信息
- **知识缺口检测**：识别缺失信息并创建针对性的子研究
- **引用管理**：跟踪并格式化所有来源的正确引用

### 💡 高级功能
- **LLM驱动分析**：使用AI理解上下文并综合信息
- **内存管理**：带缓存的研究会话持久存储
- **实时进度跟踪**：可视化界面显示实时研究进度
- **结构化报告**：生成包含清晰章节和引用的综合报告

### 🎯 核心优势
- **真正自主**：智能体独立决定下一步研究内容
- **上下文感知**：每个智能体理解整体研究目标和自身角色
- **可扩展性**：通过分解为可管理的子研究来处理复杂主题
- **透明度**：研究过程和决策制定完全可见

## 🚀 快速开始

### 前置要求

- Python 3.8 或更高版本
- [DeepSeek API Key](https://platform.deepseek.com/)（用于LLM）
- [Tavily API Key](https://tavily.com/) 或 [Serper API Key](https://serper.dev/)（用于网络搜索）

### 安装

```bash
# 克隆仓库
git clone https://github.com/phonism/research-weaver.git
cd research-weaver

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows系统使用: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
cp .env.example .env
```

编辑 `.env` 文件，添加您的API密钥：
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
# 可选：使用Serper作为备选搜索提供商
# SERPER_API_KEY=your_serper_api_key_here
```

## 📖 使用方法

### Web界面（推荐）

```bash
streamlit run ui/app.py
```

在浏览器中打开 `http://localhost:8501` 访问交互式研究界面。

### 编程使用

```python
import asyncio
from src.supervisor import SupervisorAgent
from src.tools import create_llm_client, create_tools
from src.memory_store import MemoryStore

async def main():
    # 创建组件
    llm_client = create_llm_client()
    tools = create_tools()
    memory_store = MemoryStore()
    
    # 创建监督者
    supervisor = SupervisorAgent(
        llm_client=llm_client,
        model="deepseek-chat",
        tools=tools,
        memory_store=memory_store
    )
    
    # 进行研究
    result = await supervisor.research("您的研究主题")
    print(result)

# 运行研究
asyncio.run(main())
```

### 简单API使用

```python
import asyncio
from src import research

# 使用默认设置（DeepSeek）
result = await research("量子计算的最新发展")

# 使用OpenAI
result = await research(
    "气候变化解决方案",
    provider="openai",
    api_key="your_openai_key",
    model="gpt-4-turbo-preview"
)
```

## 🔄 工作原理

### 研究流程

1. **初始规划**：监督者分析研究查询并创建研究计划
2. **智能体创建**：为不同方面创建专门的研究员智能体
3. **迭代研究**：每个智能体通过以下步骤执行专注研究：
   - **搜索**：使用搜索工具查找相关来源
   - **阅读**：从来源中提取和分析内容
   - **综合**：总结发现并识别知识缺口
   - **继续**：为缺失信息创建新的搜索查询
4. **知识整合**：监督者将所有发现整合成连贯的报告
5. **最终报告**：包含所有发现和引用的综合文档

### 智能体通信

```
用户查询 → 监督者智能体
                ↓
        创建多个研究员智能体
                ↓
    每个研究员：搜索 → 阅读 → 分析 → 报告
                ↓
        监督者整合所有发现
                ↓
            最终研究报告
```

## 🏗️ 架构

### 核心组件

- **`src/supervisor.py`**：管理研究过程的主要协调器
- **`src/researcher.py`**：具有专门焦点的个体研究智能体
- **`src/tools.py`**：用于信息收集的搜索和阅读工具
- **`src/memory_store.py`**：持久存储和缓存系统
- **`src/core.py`**：基础类和数据模型
- **`ui/app.py`**：基于Streamlit的Web界面

### 技术栈

- **LLM集成**：OpenAI兼容API（DeepSeek、OpenAI）
- **搜索API**：Tavily、Serper
- **网页抓取**：BeautifulSoup4、httpx
- **异步处理**：用于并发操作的asyncio
- **UI框架**：用于实时可视化的Streamlit
- **数据验证**：用于类型安全的Pydantic

## 🎨 界面功能

### 实时可视化
- 实时智能体状态跟踪
- 每个研究阶段的进度条
- 可展开的详细日志部分
- 引用跟踪和管理

### 交互控制
- 暂停/恢复研究功能
- 多格式导出结果
- 搜索历史和会话管理
- 可自定义的研究参数

## 🔧 配置

### 环境变量

| 变量 | 描述 | 必需 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API密钥（LLM） | 是 |
| `TAVILY_API_KEY` | Tavily搜索API密钥 | 是* |
| `SERPER_API_KEY` | Serper搜索API密钥 | 是* |
| `OPENAI_API_KEY` | OpenAI API密钥（可选） | 否 |

*需要Tavily或Serper API密钥之一

### 高级配置

```python
# 自定义LLM配置
supervisor = SupervisorAgent(
    llm_client=llm_client,
    model="deepseek-chat",
    temperature=0.7,
    max_researchers=5,
    max_rounds=3
)

# 自定义搜索参数
tools = create_tools(
    search_max_results=10,
    read_timeout=30
)
```

## 📊 研究主题示例

Research Weaver擅长处理复杂、多方面的研究主题：

- "分析可再生能源技术和市场趋势的最新发展"
- "比较临床试验中治疗阿尔茨海默病的不同方法"
- "调查AI对不同行业就业市场的经济影响"
- "研究量子计算突破的历史和现状"

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 开发设置

```bash
# 安装开发依赖
pip install -r requirements.txt

# 运行测试
pytest

# 格式化代码
black src/ ui/

# 代码检查
ruff check src/ ui/
```

## 📄 许可证

本项目基于MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

- 使用[DeepSeek](https://deepseek.com/)和[OpenAI](https://openai.com/)语言模型构建
- 搜索由[Tavily](https://tavily.com/)和[Serper](https://serper.dev/)提供支持
- UI框架由[Streamlit](https://streamlit.io/)提供

## 📮 支持

- **问题反馈**：[GitHub Issues](https://github.com/phonism/research-weaver/issues)
- **讨论**：[GitHub Discussions](https://github.com/phonism/research-weaver/discussions)

---

由Research Weaver团队用❤️制作