"""
Researcher Agent that focuses on specific research tasks with limited context
"""
import re
from typing import Optional, Dict, Any

from .core import (
    LLMAgent,
    AgentResponse,
    AgentAction,
    ActionType,
    MessageRole,
    InfoPiece
)


class ResearcherAgent(LLMAgent):
    """
    Specialized researcher agent that focuses on a single research task
    """
    
    def __init__(
        self, 
        role: str,
        initial_task: str,
        llm_client=None,
        model="gpt-4-turbo-preview",
        memory_store=None
    ):
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create a specialized system prompt based on role
        system_prompt = f"""You are a {role} researcher with deep expertise in your field.

Current Date: {current_date}

CRITICAL LANGUAGE REQUIREMENT: You MUST respond in the EXACT SAME LANGUAGE as the research task throughout the entire research process.
- If research task is in Chinese → ALL responses, analysis, and reports MUST be in Chinese
- If research task is in English → ALL responses, analysis, and reports MUST be in English  
- This includes thinking process, search queries, content analysis, and final conclusions
- NEVER switch languages unless explicitly requested

Your current research task: {initial_task}

Your approach:
1. Focus exclusively on your assigned research area
2. Search for relevant and credible information
3. Analyze and synthesize findings
4. Draw evidence-based conclusions
5. Present clear, actionable insights

You have access to the following tools:
- SEARCH: Search for information online (you can optionally use site: syntax to target specific domains)
  IMPORTANT: The search tool does NOT support OR syntax or boolean operators. Use simple keywords only.
- READ: Read content from a specific URL

RESEARCH STRATEGY: Choose ONE approach per research cycle - either SEARCH for new sources OR READ existing links.

OPTION 1 - Search for New Information:
<thinking>
[Plan what searches you need to find relevant sources and data points]
</thinking>

<action>
TOOL: SEARCH
INPUT: search query without quotes
</action>

<action>
TOOL: SEARCH
INPUT: another search query
</action>

OPTION 2 - Read Specific Sources:
<thinking>
[Decide which URLs from previous searches to read for detailed information]
</thinking>

<action>
TOOL: READ
INPUT: https://example.com/url
</action>

<action>
TOOL: READ
INPUT: https://another-url.com
</action>

CRITICAL RESEARCH LOGIC:
- DO NOT mix SEARCH and READ in the same cycle
- First cycle: SEARCH to find relevant sources
- Second cycle: READ the most promising URLs from search results
- Each cycle should focus on ONE objective: either discovering sources OR extracting content
- Do NOT put quotes around INPUT content. Write search queries or URLs directly.
- Do NOT use OR, AND, or other boolean operators in search queries. Use simple keywords only.

When you have completed your research, structure your response like this:
<thinking>
[Your final analysis and reasoning]
</thinking>

<complete>
## Research Report: {role}

### Key Findings
[Elaborate on main findings with specific data and statistics, add citations [1][2] for each important data point]

### Detailed Analysis
[In-depth analysis of research results with multiple paragraphs, each point supported by data and citations]

### Critical Data & Trends
[List important numbers, statistics, and trend information, each with citation support]

### Conclusions & Insights
[Professional conclusions and insights based on research]

## References

1. **Title** - Date - [domain.com](https://domain.com)
2. **Title** - Date - [domain.com](https://domain.com)
3. **Title** - Date - [domain.com](https://domain.com)
</complete>

IMPORTANT Research Completion Requirements: 
- Research report must be detailed and comprehensive, not just a simple list
- Every data point, opinion, and fact MUST have citations [1][2] etc
- Include specific numbers and statistics
- Provide multi-angle in-depth analysis
- Report should demonstrate professional quality with substantial valuable content

CRITICAL REFERENCE FORMATTING:
- References MUST follow EXACT format: Title - Date - domain.com
- Each reference on its own line under "References"
- Format: exactly 3 parts separated by " - "
- Extract real domain names from the sources you used
- Do NOT create fake domains or URLs
- Keep titles descriptive and accurate

CRITICAL LANGUAGE CONSISTENCY: 
- Your ENTIRE research report must be written in the SAME LANGUAGE as the research task
- If task is Chinese → report in Chinese; if English → report in English
- This includes ALL sections: findings, analysis, conclusions, and references
- Language consistency is essential for user experience

RESEARCH EFFICIENCY GUIDELINES:
- Plan comprehensive research strategy in ONE thinking session at the start
- Execute ALL planned actions together (multiple searches + reads)
- After gathering initial data, analyze results and complete research if sufficient
- Only do additional thinking+actions if there are critical information gaps
- Avoid fragmented thinking - consolidate your analysis

COMPLETION CRITERIA: Before taking any action, consider:
- Have you collected sufficient information to answer the research question?
- Are you repeating searches on similar topics?
- Do you have enough diverse sources and data points?
- If you have conducted 5+ searches and read multiple sources, complete your research.

MINIMIZE ITERATIONS: Aim to complete research in 1-2 thinking+action cycles maximum."""

        super().__init__(
            role=f"researcher_{role}",
            system_prompt=system_prompt,
            llm_client=llm_client,
            model=model,
            temperature=0.7,
            memory_store=memory_store
        )
        
        self.research_area = role
        self.initial_task = initial_task
        self.context.current_goal = initial_task
        
    def _parse_llm_output(self, output: str) -> AgentResponse:
        """
        Parse researcher's output to determine next action(s)
        """
        import logging
        logging.debug(f"[Researcher {self.research_area}] LLM Output: {output[:500]}...")
        response = AgentResponse(thought=output)
        
        # Find all action tags
        import re
        action_pattern = r'<action>(.*?)</action>'
        action_matches = re.findall(action_pattern, output, re.DOTALL)
        
        if action_matches:
            actions = []
            for action_content in action_matches:
                # Parse each action content
                lines = action_content.strip().split('\n')
                tool_line = next((l for l in lines if l.startswith('TOOL:')), None)
                input_line = next((l for l in lines if l.startswith('INPUT:')), None)
                
                if tool_line and input_line:
                    tool = tool_line.split(':', 1)[1].strip().upper()
                    input_value = input_line.split(':', 1)[1].strip()
                    
                    if tool == 'SEARCH':
                        actions.append(AgentAction(
                            action_type=ActionType.USE_TOOL,
                            parameters={"tool": "search", "query": input_value}
                        ))
                    elif tool == 'READ':
                        actions.append(AgentAction(
                            action_type=ActionType.USE_TOOL,
                            parameters={"tool": "read", "url": input_value}
                        ))
            
            # Store all actions (we'll modify AgentResponse to handle multiple actions)
            if len(actions) == 1:
                response.action = actions[0]
            elif len(actions) > 1:
                # Store multiple actions in parameters for batch execution
                response.action = AgentAction(
                    action_type=ActionType.USE_TOOL,
                    parameters={"batch_actions": actions}
                )
                    
        # Check for completion tag
        elif "<complete>" in output and "</complete>" in output:
            complete_start = output.find("<complete>") + 10
            complete_end = output.find("</complete>")
            findings = output[complete_start:complete_end].strip()
            
            response.is_complete = True
            response.result = findings  # Keep the original format with citations intact
            
        else:
            # Continue thinking if no structured output
            response.action = AgentAction(action_type=ActionType.CONTINUE_THINKING)
            
        return response
        
    async def _execute_action(self, action: AgentAction):
        """
        Execute the action(s) - single or batch with grouping by tool type
        """
        if action.action_type == ActionType.USE_TOOL:
            # Check if it's a batch of actions
            if "batch_actions" in action.parameters:
                await self._execute_batch_actions(action.parameters["batch_actions"])
            else:
                # Single action
                tool_name = action.parameters.get("tool")
                if tool_name == "search":
                    await self._execute_search(action.parameters.get("query"))
                elif tool_name == "read":
                    await self._execute_read(action.parameters.get("url"))
                
        elif action.action_type == ActionType.CONTINUE_THINKING:
            # Prompt to continue analysis
            self.context.add_message(
                MessageRole.SYSTEM,
                "Continue your research. What else do you need to investigate?"
            )
            
    async def _execute_batch_actions(self, actions: list):
        """
        Execute multiple actions sequentially for clear logging
        """
        print(f"Executing {len(actions)} actions sequentially...")
        
        for i, action in enumerate(actions, 1):
            tool = action.parameters.get("tool")
            print(f"[{i}/{len(actions)}] Executing {tool}...")
            
            if tool == "search":
                await self._execute_search(action.parameters.get("query"))
            elif tool == "read":
                await self._execute_read(action.parameters.get("url"))
                
        print("All actions completed.")
            
            
    async def _execute_search(self, query: str):
        """
        Execute a search action
        """
        if "search" not in self.available_tools:
            # Fallback to simulation if tool not available
            self.context.add_message(
                MessageRole.SYSTEM,
                "Search tool not available. Please configure tools."
            )
            return
            
        search_tool = self.available_tools["search"]
        
        try:
            # Execute actual search
            results = await search_tool.search(query, max_results=5)
            
            # Store search results in memory store for UI display
            if self.memory_store:
                self.memory_store.add_search_results(self.role, query, results)
            
            # Add citation tracking info to context
            from .citation_manager import CitationManager
            citation_manager = CitationManager()
            citation_info = citation_manager.add_citation_tracking_to_context(self.context, results)
            self.context.add_message(
                MessageRole.SYSTEM,
                citation_info
            )
            
            # Format results for LLM
            if results:
                formatted_results = f"Search results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    formatted_results += f"{i}. {result['title']}\n"
                    # Add date if available
                    if 'date' in result and result['date']:
                        formatted_results += f"   Date: {result['date']}\n"
                    formatted_results += f"   URL: {result['url']}\n"
                    formatted_results += f"   {result['snippet']}\n\n"
            else:
                formatted_results = f"No search results found for '{query}'"
            
            # Add search results to context
            import logging
            logging.debug(f"[{self.role}] Adding search results to context")
            logging.debug(f"[{self.role}] Query: {query}")
            logging.debug(f"[{self.role}] Results length: {len(formatted_results)}")
            logging.debug(f"[{self.role}] Results preview: {formatted_results[:200]}...")
            
            self.context.add_message(
                MessageRole.SYSTEM,
                formatted_results
            )
            
            logging.debug(f"[{self.role}] Context now has {len(self.context.conversation_history)} messages")
            
            # Store as collected info
            info = InfoPiece(
                content=formatted_results,
                source=f"search: {query}",
                metadata={"type": "search_results", "query": query, "results": results}
            )
            self.context.add_info(info)
            
        except Exception as e:
            self.context.add_message(
                MessageRole.SYSTEM,
                f"Error performing search: {str(e)}"
            )
        
    async def _execute_read(self, url: str):
        """
        Execute a read action
        """
        if "read" not in self.available_tools:
            # Fallback to simulation if tool not available
            self.context.add_message(
                MessageRole.SYSTEM,
                "Read tool not available. Please configure tools."
            )
            return
            
        read_tool = self.available_tools["read"]
        
        try:
            # Record read start
            if self.memory_store:
                self.memory_store.add_read_operation(self.role, url, "reading")
            
            # Execute actual read
            content = await read_tool.read_url(url)
            
            # Check if content is None
            if content is None:
                self.context.add_message(
                    MessageRole.SYSTEM,
                    f"Failed to read content from {url}"
                )
                return
            
            # If content is too long, summarize it
            if len(content) > 2000:
                # Record summarizing status
                if self.memory_store:
                    self.memory_store.add_read_operation(self.role, url, "summarizing")
                
                summary = await self._summarize_content(content, url)
                
                # Record completed with summary
                if self.memory_store:
                    self.memory_store.add_read_operation(self.role, url, "completed", summary, content)
                
                # Add summary to context instead of full content
                self.context.add_message(
                    MessageRole.SYSTEM,
                    f"Content from {url} (summarized):\n\n{summary}"
                )
                
                # Store original content as collected info
                info = InfoPiece(
                    content=content,
                    source=url,
                    metadata={"type": "article_content", "url": url, "summarized": True}
                )
                self.context.add_info(info)
            else:
                # Record completed without summary
                if self.memory_store:
                    self.memory_store.add_read_operation(self.role, url, "completed", "", content)
                
                # Add short content directly
                self.context.add_message(
                    MessageRole.SYSTEM,
                    content
                )
                
                # Store as collected info
                info = InfoPiece(
                    content=content,
                    source=url,
                    metadata={"type": "article_content", "url": url}
                )
                self.context.add_info(info)
            
        except Exception as e:
            self.context.add_message(
                MessageRole.SYSTEM,
                f"Error reading {url}: {str(e)}"
            )
        
    def _extract_key_findings(self, output: str) -> str:
        """
        Extract and format key findings from researcher's output
        """
        # Look for structured findings
        findings_start = re.search(
            r"(?:key findings|findings|conclusions?|summary):",
            output,
            re.IGNORECASE
        )
        
        if findings_start:
            findings = output[findings_start.end():].strip()
        else:
            findings = output
            
        # Format as structured findings
        formatted = f"Research Findings - {self.research_area}\n"
        formatted += f"Task: {self.initial_task}\n\n"
        formatted += findings
        
        return formatted
        
    async def research_task(self, task: str) -> str:
        """
        Execute a specific research task
        """
        self.context.current_goal = task
        self.context.add_message(MessageRole.USER, task)
        
        # Run research loop
        result = await self.run_until_complete(max_steps=20)
        
        return result
        
    async def _summarize_content(self, content: str, url: str) -> str:
        """
        Summarize long content to keep prompt manageable
        """
        try:
            # Create summarization prompt in English
            summarize_prompt = f"""Summarize the main content of this article objectively. Focus on extracting key information without adding interpretation or analysis.

Article source: {url}
Article content: {content[:3000]}...

Provide a straightforward summary of:
- Main topics and facts presented
- Key data and statistics mentioned
- Primary conclusions stated in the article

Keep the summary factual and comprehensive without length limits.

Summary:"""

            # Use LLM to summarize
            messages = [
                {"role": "system", "content": "You are a professional content summarization assistant."},
                {"role": "user", "content": summarize_prompt}
            ]
            
            import logging
            import hashlib
            import json
            
            # Generate hash for cache comparison
            messages_hash = hashlib.md5(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
            content_hash = hashlib.md5(content[:3000].encode()).hexdigest()
            
            logging.info(f"[SUMMARIZE_CALL] Agent: {self.research_area}")
            logging.info(f"[SUMMARIZE_CALL] URL: {url}")
            logging.info(f"[SUMMARIZE_CALL] Initial Task: {self.initial_task}")
            logging.info(f"[SUMMARIZE_CALL] Content Length: {len(content)}")
            logging.info(f"[SUMMARIZE_CALL] Content Hash (first 3000 chars): {content_hash}")
            logging.info(f"[SUMMARIZE_CALL] Messages Hash: {messages_hash}")
            logging.info(f"[SUMMARIZE_CALL] Model: {self.model}, Temperature: 0.3")
            logging.info(f"[SUMMARIZE_CALL] Prompt Length: {len(summarize_prompt)}")
            
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            summary = response.choices[0].message.content
            logging.info(f"[SUMMARIZE_RESPONSE] Summary Length: {len(summary)}")
            return summary
            
        except Exception as e:
            print(f"Error summarizing content: {e}")
            # Fallback: truncate content
            return content[:1500] + "...\n[Content truncated due to length]"