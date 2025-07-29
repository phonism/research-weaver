"""
Streamlit UI for Research Weaver - Real-time visualization of multi-agent research
"""
import streamlit as st
import asyncio
from datetime import datetime
import time
import os
import sys
import threading
import queue
import uuid
import re
import logging

# Setup logging for debugging
logging.basicConfig(
    filename="research_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
    encoding="utf-8"
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.supervisor import SupervisorAgent
from src.core import AgentState, MessageRole
from src.memory_store import MemoryStore


def format_llm_content(content: str, memory_store=None) -> str:
    """
    Parse LLM content and format with inline expanders for READ actions
    """
    def format_action_with_expander(match):
        action_content = match.group(1).strip()
        
        # Check if this is a READ action and get status
        if "TOOL: read" in action_content and "INPUT:" in action_content:
            lines = action_content.split("\\n")
            input_line = next((l for l in lines if l.startswith("INPUT:")), None)
            if input_line:
                url = input_line.split(":", 1)[1].strip()
                
                # Get READ status from memory store
                status = "pending"
                summary = ""
                original_content = ""
                
                if memory_store:
                    read_operations = memory_store.get_read_operations()
                    # Find the latest status for this URL
                    for read_op in reversed(read_operations):
                        if read_op["url"] == url:
                            status = read_op["status"]
                            summary = read_op.get("summary", "") or ""
                            original_content = read_op.get("original_content", "") or ""
                            break
                
                # Return formatted action with placeholder for expander
                status_text = ""
                if status == "reading":
                    status_text = " 🔄 Reading..."
                elif status == "summarizing":
                    status_text = " 🤔 Summarizing..."
                elif status == "completed":
                    status_text = " ✅ Completed"
                    if summary and len(summary) > 0:
                        status_text += " (has summary)"
                    elif original_content and len(original_content) > 0:
                        status_text += " (content too short)"
                
                # Create a unique marker for this read action
                marker = f"__READ_EXPANDER__{url}__"
                
                return f"**🔧 ACTION:** `{action_content}`{status_text}\\n{marker}"
        
        # Default action formatting
        return f"**🔧 ACTION:** `{action_content}`"
    
    # Format action tags
    formatted_content = re.sub(
        r"<action>(.*?)</action>", 
        format_action_with_expander, 
        content, 
        flags=re.DOTALL
    )
    
    # Format complete tags
    formatted_content = re.sub(
        r"<complete>(.*?)</complete>", 
        lambda m: f"\\n\\n**✅ COMPLETE:**\\n\\n{m.group(1).strip()}\\n\\n", 
        formatted_content, 
        flags=re.DOTALL
    )
    
    # Format thinking tags with special styling
    formatted_content = re.sub(
        r"<thinking>(.*?)</thinking>", 
        lambda m: f"> 🤔 **Thinking Process:**\\n> \\n> {m.group(1).strip().replace(chr(10), chr(10) + '> ')}", 
        formatted_content, 
        flags=re.DOTALL
    )
    
    return formatted_content


# Page configuration
st.set_page_config(
    page_title="Research Weaver",
    page_icon="🔬",
    layout="wide"
)


class ResearchMonitor:
    """Monitor research progress and collect updates"""
    
    def __init__(self):
        self.updates = queue.Queue()
        self.research_complete = False
        self.final_result = None
        self.error = None
        
    def add_update(self, update_type: str, data: dict):
        """Add an update to the queue"""
        self.updates.put({
            "type": update_type,
            "data": data,
            "timestamp": datetime.now()
        })
        
    def get_updates(self):
        """Get all pending updates"""
        updates = []
        while not self.updates.empty():
            try:
                updates.append(self.updates.get_nowait())
            except queue.Empty:
                break
        return updates


async def research_worker(query: str, monitor: ResearchMonitor, memory_store: MemoryStore):
    """Background worker for research"""
    try:
        # Create supervisor with DeepSeek LLM and memory store
        from src.tools import create_llm_client
        try:
            llm_client = create_llm_client()
            supervisor = SupervisorAgent(
                llm_client=llm_client, 
                model="deepseek-chat", 
                memory_store=memory_store
            )
        except ValueError as e:
            st.error(f"Configuration Error: {str(e)}")
            st.stop()
        
        # Simple monitoring hooks
        original_create_researcher = supervisor._create_researcher
        
        async def monitored_create_researcher(role: str, task: str):
            monitor.add_update("researcher_created", {"role": role, "task": task})
            result = await original_create_researcher(role, task)
            result_preview = result[:200] + "..." if len(result) > 200 else result
            monitor.add_update("researcher_completed", {"role": role, "result": result_preview})
            return result
            
        supervisor._create_researcher = monitored_create_researcher
        
        # Start research
        monitor.add_update("research_started", {"query": query})
        result = await supervisor.research(query)
        
        monitor.final_result = result
        monitor.research_complete = True
        monitor.add_update("research_completed", {"result": result})
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[Research Error] {error_msg}")
        print(f"[Research Error Trace]\\n{error_trace}")
        monitor.error = error_msg
        monitor.research_complete = True
        monitor.add_update("error", {"error": error_msg, "trace": error_trace})


def run_async_research(query: str, monitor: ResearchMonitor, memory_store: MemoryStore):
    """Run async research in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(research_worker(query, monitor, memory_store))


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "monitor" not in st.session_state:
        st.session_state.monitor = None
    if "research_thread" not in st.session_state:
        st.session_state.research_thread = None
    if "research_events" not in st.session_state:
        st.session_state.research_events = []
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = None


def render_sidebar():
    """Render the sidebar with configuration info"""
    with st.sidebar:
        st.header("Configuration")
        
        st.info("🤖 Using DeepSeek API")
        st.markdown("**Model:** DeepSeek-Chat")
        st.markdown("**Provider:** DeepSeek")
        
        st.divider()
        
        if st.button("Clear History"):
            st.session_state.research_events = []
            st.session_state.monitor = None
            st.rerun()
        
        st.markdown("""
        ### How it works
        
        1. **Supervisor** analyzes your query
        2. **Specialists** are created for specific tasks
        3. **Research** is conducted in parallel
        4. **Results** are synthesized into a report
        
        Watch the process unfold in real-time!
        """)


def render_main_input():
    """Render the main query input and start button"""
    query = st.text_area(
        "Research Query",
        value="请分析生猪市场的投资机会，包括当前价格趋势、政策变化和供需关系",
        placeholder='What would you like to research? (e.g., "Analyze investment opportunities in renewable energy")',
        height=100
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        start_button = st.button(
            "🚀 Start Research",
            disabled=bool(st.session_state.monitor and not st.session_state.monitor.research_complete),
            use_container_width=True
        )

    with col2:
        if st.session_state.monitor and not st.session_state.monitor.research_complete:
            if st.button("⏹️ Stop Research", use_container_width=True):
                st.session_state.monitor.research_complete = True
                st.warning("Research stopped by user")

    return query, start_button


def start_research_worker(query):
    """Start research in background thread"""
    import threading
    
    # Create new monitor, memory store and start research
    st.session_state.monitor = ResearchMonitor()
    st.session_state.memory_store = MemoryStore()
    st.session_state.research_events = []
    
    # Start research in background thread
    thread = threading.Thread(
        target=run_async_research,
        args=(query, st.session_state.monitor, st.session_state.memory_store)
    )
    thread.start()
    st.session_state.research_thread = thread


def render_research_progress():
    """Render the research progress UI"""
    if not st.session_state.monitor:
        return
        
    monitor = st.session_state.monitor
    
    # Get new updates
    new_updates = monitor.get_updates()
    st.session_state.research_events.extend(new_updates)
    
    # Always show LLM thinking process
    st.markdown("### 🧠 LLM Thinking Process")
    
    # Display complete LLM thinking process - read from memory store
    if st.session_state.memory_store:
        llm_responses = st.session_state.memory_store.get_llm_responses()
        if len(llm_responses) > 0:
            st.markdown("**实时AI思考过程：**")
            
            # Create a container to display all responses
            container = st.container()
            with container:
                # Display all responses in chronological order (old to new)
                for i, response in enumerate(llm_responses):
                    timestamp = response["timestamp"].strftime("%H:%M:%S")
                    agent_role = response["agent_role"]
                    content = response["response"]
                    round_num = i + 1  # Round number starts from 1
                    
                    # Format content
                    formatted_content = format_llm_content(content, st.session_state.memory_store)
                    
                    # Display title
                    if "supervisor" in agent_role.lower():
                        st.write(f"🎯 **Round {round_num} [{timestamp}] Supervisor ({agent_role}):**")
                    else:
                        st.write(f"🔍 **Round {round_num} [{timestamp}] Researcher ({agent_role}):**")
                    
                    # Process content, replace READ expander markers with actual expanders
                    lines = formatted_content.split("\\n")
                    current_block = []
                    
                    for line in lines:
                        if "__READ_EXPANDER__" in line:
                            # First display accumulated content
                            if current_block:
                                st.markdown("\\n".join(current_block))
                                current_block = []
                            
                            # Extract URL from marker and create expander
                            if line.startswith("__READ_EXPANDER__"):
                                url_from_marker = line.replace("__READ_EXPANDER__", "").replace("__", "")
                            else:
                                url_from_marker = line.replace("READ_EXPANDER__", "")
                                
                            # Find corresponding read operation from memory store
                            if st.session_state.memory_store:
                                read_operations = st.session_state.memory_store.get_read_operations()
                                for read_op in reversed(read_operations):
                                    if read_op["url"] == url_from_marker and read_op["status"] == "completed":
                                        url = read_op["url"]
                                        summary = read_op.get("summary", "") or ""
                                        original_content = read_op.get("original_content", "") or ""
                                        
                                        # Create expander
                                        if summary and len(summary) > 0:
                                            with st.expander(f"📖 View Summary - {url[:50]}...", expanded=False):
                                                st.markdown(summary)
                                                st.markdown(f"🔗 [View Original]({url})")
                                        elif original_content and len(original_content) > 0:
                                            if len(original_content) > 800:
                                                content_to_show = original_content[:800] + "...\\n[Content Truncated]"
                                            else:
                                                content_to_show = original_content
                                            with st.expander(f"📄 View Content - {url[:50]}...", expanded=False):
                                                st.text(content_to_show)
                                                st.markdown(f"🔗 [View Original]({url})")
                                        break
                        else:
                            current_block.append(line)
                    
                    # Display remaining content
                    if current_block:
                        st.markdown("\\n".join(current_block))
                    
                    st.write("---")
        else:
            st.write("Waiting for AI to start thinking...")


def render_search_results():
    """Render search results section"""
    if not st.session_state.memory_store:
        return
        
    search_results = st.session_state.memory_store.get_search_results()
    if not search_results:
        return
        
    st.markdown("### 🔍 Search Results")
    
    for search_data in search_results:
        timestamp = search_data["timestamp"].strftime("%H:%M:%S")
        agent_role = search_data["agent_role"]
        query = search_data["query"]
        results = search_data["results"]
        
        with st.expander(f"[{timestamp}] {agent_role} Search: {query}", expanded=False):
            if results:
                for i, result in enumerate(results, 1):
                    st.markdown(f"**{i}. {result['title']}**")
                    st.markdown(f"🔗 {result['url']}")
                    if result['snippet']:
                        st.markdown(f"📝 {result['snippet']}")
                    st.markdown("---")
            else:
                st.write("No search results found")


def _process_citations_for_display(text: str) -> str:
    """
    Process citations in text to create clickable superscript links
    """
    import re
    
    # Find all citations in format [1], [2], etc.
    citations = re.findall(r'\\[(\\d+)\\]', text)
    
    if not citations:
        return text
    
    # Replace citations in main text (but not in references section) with clickable superscript links
    def replace_citation(match):
        citation_num = match.group(1)
        return f'<sup><a href="#ref-{citation_num}" style="text-decoration: none; color: #1f77b4; font-weight: bold;">[{citation_num}]</a></sup>'
    
    # Split text into parts before and after references section
    ref_patterns = [r'## References', r'## 参考文献', r'参考文献(?!\\s*\\w)', r'References(?!\\s*\\w)']
    
    ref_section_start = -1
    ref_section_pattern = None
    
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            ref_section_start = match.start()
            ref_section_pattern = pattern
            break
    
    if ref_section_start >= 0:
        # Split text into main content and references
        main_content = text[:ref_section_start]
        ref_section = text[ref_section_start:]
        
        # Process citations only in main content
        processed_main = re.sub(r'\\[(\\d+)\\]', replace_citation, main_content)
        
        # Add anchors to numbered references in reference section
        def replace_reference_flexible(match):
            # Handle both "1. Title" and "[1] Title" formats
            ref_num = match.group(1) or match.group(2)  # Either capture group
            rest = match.group(3)
            
            # Convert markdown links to HTML links
            rest_with_html_links = re.sub(
                r'\\[([^\\]]+)\\]\\(([^)]+)\\)',
                r'<a href="\\2" target="_blank" style="color: #1f77b4; text-decoration: underline;">\\1</a>',
                rest
            )
            
            # Add anchor with proper HTML structure
            return f'<div id="ref-{ref_num}" style="margin-bottom: 10px; padding: 8px; border-left: 3px solid #1f77b4; background-color: #f8f9fa; border-radius: 4px;"><strong>{ref_num}.</strong> {rest_with_html_links}</div>'
        
        # Match numbered references at start of line in reference section
        # Support both formats: "1. Title" and "[1] Title"
        processed_refs = re.sub(r'^(?:(\\d+)\\.|\\[(\\d+)\\])\\s*(.+)$', 
                               replace_reference_flexible, 
                               ref_section, flags=re.MULTILINE)
        
        return processed_main + processed_refs
    else:
        # No references section found, just process citations
        return re.sub(r'\\[(\\d+)\\]', replace_citation, text)


def render_final_result():
    """Render the final research result"""
    if not (st.session_state.monitor and st.session_state.monitor.research_complete and st.session_state.monitor.final_result):
        return
        
    monitor = st.session_state.monitor
    
    st.header("📄 Research Report")
    
    # Process citations for clickable links
    processed_result = _process_citations_for_display(monitor.final_result)
    
    # Add CSS for better citation styling
    citation_css = """
    <style>
    sup a {
        transition: all 0.2s ease;
    }
    sup a:hover {
        background-color: #e6f3ff;
        border-radius: 3px;
        padding: 1px 2px;
    }
    div[id^="ref-"]:target {
        background-color: #fff3cd !important;
        border-left-color: #ffc107 !important;
        animation: highlight 2s ease-out;
    }
    div[id^="ref-"] a {
        color: #1f77b4;
        text-decoration: underline;
    }
    div[id^="ref-"] a:hover {
        color: #0d47a1;
        text-decoration: underline;
    }
    @keyframes highlight {
        0% { background-color: #fff3cd; }
        100% { background-color: #f8f9fa; }
    }
    </style>
    """
    
    # Display CSS and content
    st.markdown(citation_css, unsafe_allow_html=True)
    st.markdown(processed_result, unsafe_allow_html=True)
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=monitor.final_result,  # Keep original format for download
        file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


def render_welcome_message():
    """Render welcome message for new users"""
    if st.session_state.monitor:
        return
        
    st.info("""
    👋 **Welcome to Research Weaver!**
    
    1. Set your `DEEPSEEK_API_KEY` and `TAVILY_API_KEY` environment variables
    2. Modify the research question above
    3. Click "Start Research" button to start
    4. Watch how AI agents collaborate to conduct deep research
    
    Powered by DeepSeek AI and Tavily Search!
    """)


def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🔬 Research Weaver")
    st.markdown("AI-powered multi-agent research system with real-time monitoring")

    # Sidebar
    render_sidebar()

    # Main content
    query, start_button = render_main_input()

    # Handle start button
    if start_button and query:
        start_research_worker(query)

    # Display research progress
    render_research_progress()
    
    # Display search results
    render_search_results()
    
    # Auto-refresh while research is active
    if st.session_state.monitor and not st.session_state.monitor.research_complete:
        time.sleep(0.5)
        st.rerun()
    
    # Display final result
    render_final_result()
    
    # Handle errors
    if st.session_state.monitor and st.session_state.monitor.error:
        st.error(f"Research failed: {st.session_state.monitor.error}")
    
    # Welcome message
    render_welcome_message()
    
    # Footer
    st.divider()
    st.caption("Research Weaver - Multi-Agent AI Research System")


if __name__ == "__main__":
    main()