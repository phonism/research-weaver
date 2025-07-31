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
import re
import logging

# Setup logging for debugging
logging.basicConfig(
    filename="research_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
    encoding="utf-8",
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.supervisor import SupervisorAgent
from src.memory_store import MemoryStore
from .utils import calculate_token_cost, format_cost_display, format_token_display, get_status_emoji, format_cost_breakdown


def format_llm_content(content: str, memory_store=None) -> str:
    """
    Parse LLM content and format with inline expanders for READ actions
    """

    def format_action_with_expander(match):
        action_content = match.group(1).strip()

        # Check if this is a READ action and get status (case insensitive)
        if ("TOOL: read" in action_content or "TOOL: READ" in action_content) and "INPUT:" in action_content:
            lines = action_content.split("\n")
            input_line = next((line for line in lines if line.startswith("INPUT:")), None)
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

                return f"**🔧 ACTION:** `{action_content}`{status_text}\n{marker}"

        # Default action formatting
        return f"**🔧 ACTION:** `{action_content}`"

    # Format action tags
    formatted_content = re.sub(r"<action>(.*?)</action>", format_action_with_expander, content, flags=re.DOTALL)

    # Format complete tags
    formatted_content = re.sub(
        r"<complete>(.*?)</complete>",
        lambda m: f"\n\n**✅ COMPLETE:**\n\n{m.group(1).strip()}\n\n",
        formatted_content,
        flags=re.DOTALL,
    )

    # Format thinking tags with special styling
    formatted_content = re.sub(
        r"<thinking>(.*?)</thinking>",
        lambda m: f"> 🤔 **Thinking Process:**\n> \n> {m.group(1).strip().replace(chr(10), chr(10) + '> ')}",
        formatted_content,
        flags=re.DOTALL,
    )

    return formatted_content


# Page configuration
st.set_page_config(page_title="Research Weaver", page_icon="🔬", layout="wide")


class ResearchMonitor:
    """Monitor research progress and collect updates"""

    def __init__(self):
        self.updates = queue.Queue()
        self.research_complete = False
        self.final_result = None
        self.error = None
        # Enhanced status tracking
        self.current_status = "idle"
        self.current_task = ""
        self.current_researcher = None
        self.total_searches = 0
        self.total_reads = 0
        self.pending_reads = []
        self.completed_reads = []
        self.last_activity_time = datetime.now()
        self.current_llm_request = None
        self.llm_request_start_time = None
        self.total_llm_requests = 0
        self.total_rounds = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add_update(self, update_type: str, data: dict):
        """Add an update to the queue"""
        self.updates.put({"type": update_type, "data": data, "timestamp": datetime.now()})
        self.last_activity_time = datetime.now()
        
        # Update status based on update type
        if update_type == "research_started":
            self.current_status = "initializing"
        elif update_type == "researcher_created":
            self.current_status = "researching"
            self.current_researcher = data.get("role", "")
            self.current_task = data.get("task", "")
            self.total_rounds += 1
        elif update_type == "search_started":
            self.current_status = "searching"
            self.total_searches += 1
        elif update_type == "read_started":
            self.current_status = "reading"
            self.total_reads += 1
            self.pending_reads.append(data.get("url", ""))
        elif update_type == "read_completed":
            url = data.get("url", "")
            if url in self.pending_reads:
                self.pending_reads.remove(url)
            self.completed_reads.append(url)
            if not self.pending_reads:
                self.current_status = "processing"
        elif update_type == "researcher_completed":
            self.current_status = "synthesizing"
        elif update_type == "research_completed":
            self.current_status = "completed"
        elif update_type == "error":
            self.current_status = "error"
        elif update_type == "llm_request_started":
            self.current_status = "thinking"
            self.current_llm_request = data.get("prompt_preview", "")
            self.llm_request_start_time = datetime.now()
            self.total_llm_requests += 1
        elif update_type == "llm_request_completed":
            if self.current_status == "thinking":
                self.current_status = "processing"
            self.current_llm_request = None
            self.llm_request_start_time = None
            # Add token counts if available
            if "input_tokens" in data:
                self.total_input_tokens += data["input_tokens"]
            if "output_tokens" in data:
                self.total_output_tokens += data["output_tokens"]
        elif update_type == "llm_request_error":
            self.current_status = "error"
            self.current_llm_request = None
            self.llm_request_start_time = None

    def get_updates(self):
        """Get all pending updates"""
        updates = []
        while not self.updates.empty():
            try:
                updates.append(self.updates.get_nowait())
            except queue.Empty:
                break
        return updates
    
    def get_status_info(self):
        """Get current status information"""
        time_since_activity = (datetime.now() - self.last_activity_time).total_seconds()
        
        # Calculate LLM request duration if in progress
        llm_request_duration = None
        if self.llm_request_start_time:
            llm_request_duration = (datetime.now() - self.llm_request_start_time).total_seconds()
        
        return {
            "status": self.current_status,
            "researcher": self.current_researcher,
            "task": self.current_task,
            "searches": self.total_searches,
            "reads": self.total_reads,
            "pending_reads": len(self.pending_reads),
            "completed_reads": len(self.completed_reads),
            "time_since_activity": time_since_activity,
            "total_llm_requests": self.total_llm_requests,
            "current_llm_request": self.current_llm_request,
            "llm_request_duration": llm_request_duration,
            "is_stuck": time_since_activity > 120 and self.current_status not in ["completed", "error", "idle"],  # 增加到2分钟
            "is_thinking": self.current_status == "thinking",
            "total_rounds": self.total_rounds,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens
        }


async def research_worker(query: str, monitor: ResearchMonitor, memory_store: MemoryStore):
    """Background worker for research"""
    try:
        # Create supervisor with DeepSeek LLM and memory store
        from src.tools import create_llm_client

        try:
            llm_client = create_llm_client()
            supervisor = SupervisorAgent(llm_client=llm_client, model="deepseek-chat", memory_store=memory_store)
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
        
        # Add monitoring directly to memory store callbacks
        if memory_store:
            # Monitor search operations
            original_add_search = memory_store.add_search_results
            def monitored_add_search(agent_role: str, query: str, results: list):
                print(f"[MONITOR] Search detected: {query} by {agent_role}")
                monitor.add_update("search_started", {"query": query})
                monitor.add_update("search_completed", {"query": query, "results": len(results)})
                return original_add_search(agent_role, query, results)
            memory_store.add_search_results = monitored_add_search
            
            # Monitor read operations  
            original_add_read = memory_store.add_read_operation
            def monitored_add_read(agent_role: str, url: str, status: str, summary: str = "", original_content: str = ""):
                if status == "reading":
                    monitor.add_update("read_started", {"url": url})
                elif status == "completed":
                    monitor.add_update("read_completed", {"url": url})
                elif status == "error":
                    monitor.add_update("read_error", {"url": url})
                return original_add_read(agent_role, url, status, summary, original_content)
            memory_store.add_read_operation = monitored_add_read
            
            # Monitor LLM requests by patching the adapter generate method
            # Check if using internal LLM
            try:
                from src.internal_llm import InternalLLMAdapter
                # If internal LLM is available, patch it
                original_internal_generate = InternalLLMAdapter.generate
                
                async def monitored_internal_generate(self, messages, model="deepseek-v3", temperature=0.7):
                    # Extract prompt preview from messages
                    prompt_preview = ""
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        if "content" in last_message:
                            content = last_message["content"]
                            prompt_preview = content[:200] + "..." if len(content) > 200 else content
                    
                    print(f"[MONITOR] Internal LLM request started")
                    monitor.add_update("llm_request_started", {"prompt_preview": prompt_preview, "agent": "internal"})
                    
                    try:
                        result = await original_internal_generate(self, messages, model, temperature)
                        print(f"[MONITOR] Internal LLM request completed")
                        monitor.add_update("llm_request_completed", {"agent": "internal"})
                        return result
                    except Exception as e:
                        print(f"[MONITOR] Internal LLM request error: {e}")
                        monitor.add_update("llm_request_error", {"agent": "internal", "error": str(e)})
                        raise
                
                InternalLLMAdapter.generate = monitored_internal_generate
                print("[MONITOR] Patched InternalLLMAdapter.generate")
                
            except ImportError:
                print("[MONITOR] InternalLLMAdapter not available, trying standard adapters")
                
                # Fallback to standard adapters
                from src.llm_adapter import DeepSeekAdapter, OpenAIAdapter
                
                # Patch DeepSeek adapter
                original_deepseek_generate = DeepSeekAdapter.generate
                async def monitored_deepseek_generate(self, messages, model="deepseek-chat", temperature=0.7):
                    prompt_preview = ""
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        if "content" in last_message:
                            content = last_message["content"]
                            prompt_preview = content[:200] + "..." if len(content) > 200 else content
                    
                    print(f"[MONITOR] DeepSeek LLM request started")
                    monitor.add_update("llm_request_started", {"prompt_preview": prompt_preview, "agent": "deepseek"})
                    
                    try:
                        result = await original_deepseek_generate(self, messages, model, temperature)
                        print(f"[MONITOR] DeepSeek LLM request completed")
                        monitor.add_update("llm_request_completed", {"agent": "deepseek"})
                        return result
                    except Exception as e:
                        print(f"[MONITOR] DeepSeek LLM request error: {e}")
                        monitor.add_update("llm_request_error", {"agent": "deepseek", "error": str(e)})
                        raise
                        
                DeepSeekAdapter.generate = monitored_deepseek_generate
                print("[MONITOR] Patched DeepSeekAdapter.generate")

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
        print(f"[Research Error Trace]\n{error_trace}")
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
        st.header("⚙️ Configuration")

        st.info("🤖 Using DeepSeek API")
        st.markdown("**Model:** DeepSeek-Chat")
        st.markdown("**Provider:** DeepSeek")

        st.divider()

        # Report formatting options
        st.subheader("📄 Report Settings")

        # Initialize session state for report settings
        if "report_add_metadata" not in st.session_state:
            st.session_state.report_add_metadata = True
        if "report_add_toc" not in st.session_state:
            st.session_state.report_add_toc = True
        if "report_add_icons" not in st.session_state:
            st.session_state.report_add_icons = True

        st.session_state.report_add_metadata = st.checkbox(
            "Add metadata section",
            value=st.session_state.report_add_metadata,
            help="Include generation time and system info",
        )

        st.session_state.report_add_toc = st.checkbox(
            "Add table of contents", value=st.session_state.report_add_toc, help="Generate TOC from section headers"
        )

        st.session_state.report_add_icons = st.checkbox(
            "Add section icons", value=st.session_state.report_add_icons, help="Add visual icons to section headers"
        )

        st.divider()

        if st.button("🗑️ Clear History"):
            st.session_state.research_events = []
            st.session_state.monitor = None
            st.rerun()

        st.markdown(
            """
        ### 🔬 How it works
        
        1. **Supervisor** analyzes your query
        2. **Specialists** are created for specific tasks  
        3. **Research** is conducted in parallel
        4. **Results** are synthesized into a report
        
        Watch the process unfold in real-time!
        
        ### 📊 Export Features
        
        - **Enhanced Markdown**: TOC, icons, metadata
        - **Professional HTML**: Styled, interactive
        - **Dark Theme HTML**: Perfect for presentations
        - **Citations**: Clickable references
        """
        )


def render_main_input():
    """Render the main query input and start button"""
    query = st.text_area(
        "Research Query",
        value="请分析生猪市场的投资机会，包括当前价格趋势、政策变化和供需关系",
        placeholder='What would you like to research? (e.g., "Analyze investment opportunities in renewable energy")',
        height=100,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        start_button = st.button(
            "🚀 Start Research",
            disabled=bool(st.session_state.monitor and not st.session_state.monitor.research_complete),
            use_container_width=True,
        )

    with col2:
        if st.session_state.monitor and not st.session_state.monitor.research_complete:
            if st.button("⏹️ Stop Research", use_container_width=True):
                st.session_state.monitor.research_complete = True
                st.warning("Research stopped by user")

    return query, start_button


def start_research_worker(query, research_func=None):
    """Start research in background thread"""

    # Use provided research function or default
    if research_func is None:
        research_func = run_async_research

    # Create new monitor, memory store and start research
    st.session_state.monitor = ResearchMonitor()
    st.session_state.memory_store = MemoryStore()
    st.session_state.research_events = []

    # Start research in background thread
    thread = threading.Thread(
        target=research_func, args=(query, st.session_state.monitor, st.session_state.memory_store)
    )
    thread.start()
    st.session_state.research_thread = thread


def render_status_monitor():
    """Render real-time status monitoring panel"""
    if not st.session_state.monitor:
        return
    
    monitor = st.session_state.monitor
    status_info = monitor.get_status_info()
    
    # Create two rows of metrics for better space utilization
    # First row: Status, Rounds, Searches, Reads, LLM Calls
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns([2.5, 1, 1, 1, 1.5])
    # Second row: Tokens and Cost with more space
    row2_col1, row2_col2 = st.columns([3, 2])
    
    # First row
    with row1_col1:
        status_emoji = get_status_emoji(status_info["status"])
        
        # Show thinking duration if currently thinking
        if status_info["is_thinking"] and status_info["llm_request_duration"]:
            duration_text = f"({int(status_info['llm_request_duration'])}s)"
            st.metric("Status", f"{status_emoji} {status_info['status'].capitalize()}", duration_text)
        else:
            st.metric("Status", f"{status_emoji} {status_info['status'].capitalize()}")
    
    with row1_col2:
        st.metric("Rounds", status_info["total_rounds"])
        
    with row1_col3:
        st.metric("Searches", status_info["searches"])
        
    with row1_col4:
        st.metric("Reads", status_info["reads"])
        
    with row1_col5:
        st.metric("LLM Calls", status_info["total_llm_requests"])
    
    # Second row    
    with row2_col1:
        token_display = format_token_display(status_info["total_input_tokens"], status_info["total_output_tokens"])
        st.metric("Tokens", token_display)
    
    with row2_col2:
        costs = calculate_token_cost(status_info["total_input_tokens"], status_info["total_output_tokens"])
        cost_display = format_cost_display(costs["total_cost"])
        st.metric("Cost", cost_display)
    
    # Show current task info if available
    if status_info["researcher"]:
        st.info(f"**Current Researcher:** {status_info['researcher']}")
        if status_info["task"]:
            st.caption(f"Task: {status_info['task'][:200]}...")
    
    # Show current thinking state if LLM is processing
    if status_info["is_thinking"]:
        thinking_duration = int(status_info["llm_request_duration"]) if status_info["llm_request_duration"] else 0
        st.info(f"🧠 **AI is thinking...** ({thinking_duration}s)")
        if status_info["current_llm_request"]:
            with st.expander("Current Request Preview", expanded=False):
                st.text(status_info["current_llm_request"])
    
    # Show pending reads if any
    if status_info["pending_reads"]:
        with st.expander(f"⏳ Pending Reads ({status_info['pending_reads']})", expanded=False):
            for url in monitor.pending_reads:
                st.caption(f"• {url}")
    
    # Show completed reads if any
    if status_info["completed_reads"]:
        with st.expander(f"✅ Completed Reads ({status_info['completed_reads']})", expanded=False):
            for url in monitor.completed_reads[-5:]:  # Show last 5
                st.caption(f"• {url}")
    
    # Warning if stuck - now collapsible and less intrusive
    if status_info["is_stuck"]:
        if status_info["is_thinking"]:
            warning_msg = f"⚠️ AI has been thinking for {int(status_info['llm_request_duration'])}s. This might indicate a slow LLM response."
        else:
            warning_msg = "⚠️ Research appears to be stuck. It's been over 2 minutes since the last activity."
        
        # Show as collapsible expander instead of prominent warning
        with st.expander("⚠️ Potential Issue Detected", expanded=False):
            st.warning(warning_msg)
            
            # Debug info inside the expander
            st.write(f"**Current Status:** {status_info['status']}")
            st.write(f"**Last Activity:** {status_info['time_since_activity']:.1f} seconds ago")
            st.write(f"**Current Researcher:** {status_info['researcher']}")
            st.write(f"**Total LLM Requests:** {status_info['total_llm_requests']}")
            st.write(f"**Total Searches:** {status_info['searches']}")
            st.write(f"**Reads:** {status_info['completed_reads']}/{status_info['reads']}")
            
            # Cost breakdown
            cost_breakdown = format_cost_breakdown(status_info["total_input_tokens"], status_info["total_output_tokens"])
            st.write(cost_breakdown)
            
            if status_info["is_thinking"]:
                st.error(f"🧠 **STUCK IN LLM REQUEST** - Duration: {int(status_info['llm_request_duration'])}s")
                if status_info["current_llm_request"]:
                    st.text_area("Current LLM Request", status_info["current_llm_request"], height=150)
                st.markdown("**Possible causes:**")
                st.markdown("- LLM API is slow or overloaded")
                st.markdown("- Network connection issues")
                st.markdown("- Complex request requiring long processing")
            else:
                st.error("❓ **STUCK IN UNKNOWN STATE**")
                st.markdown("**Possible causes:**")
                st.markdown("- Code execution stopped unexpectedly")
                st.markdown("- Waiting for external resource")
                st.markdown("- Internal error not caught")
            
            # Show last few log entries
            if hasattr(st.session_state.memory_store, 'get_llm_responses'):
                llm_responses = st.session_state.memory_store.get_llm_responses()
                if llm_responses:
                    last_response = llm_responses[-1]
                    st.write(f"**Last LLM Response Time:** {last_response['timestamp']}")
                    st.write(f"**Last Agent:** {last_response['agent_role']}")
                    st.text_area("Last Response Preview", last_response['response'][:500], height=200)
    
    st.divider()

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
                    lines = formatted_content.split("\n")
                    current_block = []

                    for line in lines:
                        if "__READ_EXPANDER__" in line:
                            # First display accumulated content
                            if current_block:
                                st.markdown("\n".join(current_block))
                                current_block = []

                            # Extract URL from marker and create expander
                            # Marker format: __READ_EXPANDER__URL__
                            if line.startswith("__READ_EXPANDER__"):
                                # Remove prefix and suffix
                                url_from_marker = line[len("__READ_EXPANDER__") :]
                                if url_from_marker.endswith("__"):
                                    url_from_marker = url_from_marker[:-2]
                            else:
                                # Handle lines that contain the marker but don't start with it
                                marker_start = line.find("__READ_EXPANDER__")
                                if marker_start >= 0:
                                    # Find the ending __ after the URL
                                    url_part = line[marker_start + len("__READ_EXPANDER__") :]
                                    if url_part.endswith("__"):
                                        url_from_marker = url_part[:-2]
                                    else:
                                        url_from_marker = url_part
                                else:
                                    # Fallback - shouldn't happen
                                    url_from_marker = line.replace("READ_EXPANDER__", "")

                            # Find corresponding read operation from memory store
                            if st.session_state.memory_store:
                                read_operations = st.session_state.memory_store.get_read_operations()

                                # Debug logging for URL matching
                                import logging

                                logging.debug(f"[UI_DEBUG] Looking for URL: {url_from_marker}")
                                logging.debug(f"[UI_DEBUG] Available read operations: {len(read_operations)}")
                                for i, op in enumerate(read_operations):
                                    logging.debug(f"[UI_DEBUG] Read op {i}: {op['url']} - {op['status']}")

                                for read_op in reversed(read_operations):
                                    if read_op["url"] == url_from_marker and read_op["status"] == "completed":
                                        url = read_op["url"]
                                        summary = read_op.get("summary", "") or ""
                                        original_content = read_op.get("original_content", "") or ""

                                        # Debug logging
                                        import logging

                                        logging.debug(f"[UI_DEBUG] Found completed read for {url}")
                                        logging.debug(f"[UI_DEBUG] Summary length: {len(summary)}")
                                        logging.debug(f"[UI_DEBUG] Content length: {len(original_content)}")

                                        # Create expander - prioritize summary if available, otherwise show content
                                        content_available = False
                                        if summary and len(summary.strip()) > 0:
                                            with st.expander(f"📖 View Summary - {url[:50]}...", expanded=False):
                                                st.markdown(summary)
                                                st.markdown(f"🔗 [View Original]({url})")
                                            content_available = True
                                        elif original_content and len(original_content.strip()) > 0:
                                            if len(original_content) > 800:
                                                content_to_show = original_content[:800] + "...\n[Content Truncated]"
                                            else:
                                                content_to_show = original_content
                                            with st.expander(f"📄 View Content - {url[:50]}...", expanded=False):
                                                st.text(content_to_show)
                                                st.markdown(f"🔗 [View Original]({url})")
                                            content_available = True

                                        if not content_available:
                                            # Show a placeholder if no content available
                                            with st.expander(f"⚠️ No Content - {url[:50]}...", expanded=False):
                                                st.warning("Content could not be loaded or was empty")
                                                st.markdown(f"🔗 [View Original]({url})")

                                        break
                        else:
                            current_block.append(line)

                    # Display remaining content
                    if current_block:
                        st.markdown("\n".join(current_block))

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
                    if result["snippet"]:
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
    citations = re.findall(r"\\[(\\d+)\\]", text)

    if not citations:
        return text

    # Replace citations in main text (but not in references section) with clickable superscript links
    def replace_citation(match):
        citation_num = match.group(1)
        return f'<sup><a href="#ref-{citation_num}" style="text-decoration: none; color: #1f77b4; font-weight: bold;">[{citation_num}]</a></sup>'

    # Split text into parts before and after references section
    ref_patterns = [r"## References", r"## 参考文献", r"参考文献(?!\\s*\\w)", r"References(?!\\s*\\w)"]

    ref_section_start = -1

    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            ref_section_start = match.start()
            break

    if ref_section_start >= 0:
        # Split text into main content and references
        main_content = text[:ref_section_start]
        ref_section = text[ref_section_start:]

        # Process citations only in main content
        processed_main = re.sub(r"\\[(\\d+)\\]", replace_citation, main_content)

        # Add anchors to numbered references in reference section
        def replace_reference_flexible(match):
            # Handle both "1. Title" and "[1] Title" formats
            ref_num = match.group(1) or match.group(2)  # Either capture group
            rest = match.group(3)

            # Convert markdown links to HTML links
            rest_with_html_links = re.sub(
                r"\\[([^\\]]+)\\]\\(([^)]+)\\)",
                r'<a href="\\2" target="_blank" style="color: #1f77b4; text-decoration: underline;">\\1</a>',
                rest,
            )

            # Add anchor with proper HTML structure
            return f'<div id="ref-{ref_num}" style="margin-bottom: 10px; padding: 8px; border-left: 3px solid #1f77b4; background-color: #f8f9fa; border-radius: 4px;"><strong>{ref_num}.</strong> {rest_with_html_links}</div>'

        # Match numbered references at start of line in reference section
        # Support both formats: "1. Title" and "[1] Title"
        processed_refs = re.sub(
            r"^(?:(\\d+)\\.|\\[(\\d+)\\])\\s*(.+)$", replace_reference_flexible, ref_section, flags=re.MULTILINE
        )

        return processed_main + processed_refs
    else:
        # No references section found, just process citations
        return re.sub(r"\\[(\\d+)\\]", replace_citation, text)


def render_final_result():
    """Render the final research result"""
    if not (
        st.session_state.monitor
        and st.session_state.monitor.research_complete
        and st.session_state.monitor.final_result
    ):
        return

    monitor = st.session_state.monitor

    # Import and initialize report formatter
    from .report_formatter import ReportFormatter

    formatter = ReportFormatter()

    # Process citations for clickable links
    _process_citations_for_display(monitor.final_result)

    # Add CSS for better citation styling

    # Preview section
    st.markdown("### 📄 Report Preview")

    # Generate export options with user settings
    report_title = "Research Report"

    # Apply user settings for enhanced markdown
    enhanced_content = monitor.final_result
    if st.session_state.get("report_add_icons", True):
        enhanced_content = formatter._enhance_sections(enhanced_content)

    # Create tabs for different previews
    tab1, tab2, tab3 = st.tabs(["🌐 HTML Light", "🌙 HTML Dark", "📄 Markdown"])

    with tab1:
        st.markdown("**Professional HTML Theme:**")
        # Use complete report content without truncation
        html_preview = formatter.format_html_report(enhanced_content, "Research Report", "professional")
        st.components.v1.html(html_preview, height=800, scrolling=True)

    with tab2:
        st.markdown("**Dark Theme HTML:**")
        # Use complete report content without truncation
        html_dark_preview = formatter.format_html_report(enhanced_content, "Research Report", "dark")
        st.components.v1.html(html_dark_preview, height=800, scrolling=True)

    with tab3:
        st.markdown("**Enhanced Markdown with TOC and Icons:**")
        preview_content = enhanced_content
        if st.session_state.get("report_add_metadata", True):
            preview_content = formatter._generate_metadata() + "\n\n" + preview_content
        if st.session_state.get("report_add_toc", True):
            preview_content = formatter._add_table_of_contents(preview_content)
        # Show complete content as rendered markdown
        # Use a container with custom styling to ensure proper markdown rendering
        with st.container():
            st.markdown(preview_content, unsafe_allow_html=True)

    st.markdown("---")

    # Download section - bottom of the layout
    st.markdown("### 📥 Download Options")

    # Generate export options using the formatter
    export_options = formatter.generate_export_options(enhanced_content, report_title)
    
    # Determine number of columns based on available options
    num_columns = len(export_options)
    if num_columns == 3:
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
    elif num_columns == 4:
        col1, col2, col3, col4 = st.columns(4)
        columns = [col1, col2, col3, col4]
    else:
        columns = st.columns(num_columns)
    
    # Create download buttons for each export option
    button_labels = {
        "markdown": "📄 Download Markdown",
        "html": "🌐 Download HTML",
        "html_dark": "🌙 Download HTML (Dark)",
        "pdf": "📑 Download PDF"
    }
    
    for i, (option_key, option_data) in enumerate(export_options.items()):
        if i < len(columns):
            with columns[i]:
                # Generate timestamped filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"research_report_{timestamp}.{option_key.split('_')[0] if '_' not in option_key else option_key.replace('_', '_')}"
                if option_key == "html_dark":
                    filename = f"research_report_dark_{timestamp}.html"
                elif option_key == "pdf":
                    filename = f"research_report_{timestamp}.pdf"
                
                st.download_button(
                    label=button_labels.get(option_key, f"📄 Download {option_key.title()}"),
                    data=option_data["content"],
                    file_name=filename,
                    mime=option_data["mime_type"],
                    use_container_width=True,
                )


def render_welcome_message():
    """Render welcome message for new users"""
    if st.session_state.monitor:
        return

    st.info(
        """
    👋 **Welcome to Research Weaver!**
    
    1. Set your `DEEPSEEK_API_KEY` and `TAVILY_API_KEY` environment variables
    2. Modify the research question above
    3. Click "Start Research" button to start
    4. Watch how AI agents collaborate to conduct deep research
    
    Powered by DeepSeek AI and Tavily Search!
    """
    )


def main(research_func=None):
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()

    # Main title
    st.title("🔬 Research Weaver")
    st.markdown("AI-powered multi-agent research system with real-time monitoring")

    # Sidebar
    render_sidebar()

    # Main content - always show input at top
    query, start_button = render_main_input()

    # Handle start button
    if start_button and query:
        start_research_worker(query, research_func)

    # Display final result after input if available
    if (
        st.session_state.monitor
        and st.session_state.monitor.research_complete
        and st.session_state.monitor.final_result
    ):
        st.divider()
        render_final_result()

    # Handle errors
    if st.session_state.monitor and st.session_state.monitor.error:
        st.error(f"Research failed: {st.session_state.monitor.error}")

    # Display research progress and status monitor
    if st.session_state.monitor:
        # Always show status monitor when there's an active session
        st.markdown("### 📊 Research Status Monitor")
        render_status_monitor()
        
        if not st.session_state.monitor.research_complete:
            # Show expanded progress for ongoing research
            render_research_progress()
            render_search_results()
        else:
            # Show collapsed progress for completed research
            with st.expander("🔍 View Research Process", expanded=False):
                render_research_progress()
                render_search_results()

    # Auto-refresh while research is active
    if st.session_state.monitor and not st.session_state.monitor.research_complete:
        time.sleep(0.5)
        st.rerun()

    # Welcome message (only show if no research has been done)
    if not st.session_state.monitor:
        render_welcome_message()
    
    # Footer
    st.divider()
    st.caption("Research Weaver - Multi-Agent AI Research System")

    # Quick Navigation Buttons - Always visible
    st.markdown("""
    <style>
    .nav-buttons {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .nav-btn {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        text-decoration: none;
    }
    .nav-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .nav-btn:active {
        transform: translateY(0px);
    }
    .to-top::before {
        content: "▲";
    }
    .to-bottom::before {
        content: "▼";
    }
    </style>
    
    <div class="nav-buttons">
        <a href="#" class="nav-btn to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'}); return false;" title="回到顶部"></a>
        <a href="#" class="nav-btn to-bottom" onclick="window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'}); return false;" title="快速到底部"></a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
