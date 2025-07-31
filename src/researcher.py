"""
Researcher Agent that focuses on specific research tasks with limited context
"""

import re
import logging
from datetime import datetime

from .core import LLMAgent, AgentResponse, AgentAction, ActionType, MessageRole, InfoPiece


class ResearcherAgent(LLMAgent):
    """
    Specialized researcher agent that focuses on a single research task
    """

    def __init__(self, role: str, initial_task: str, llm_client=None, model="gpt-4-turbo-preview", memory_store=None):
        current_date = datetime.now().strftime("%Y-%m-%d")

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
1. Understand the depth and scope your research task requires
2. Search for relevant and credible information appropriate to the task complexity
3. Analyze and synthesize findings with the depth the topic deserves
4. Draw evidence-based conclusions suited to the research context
5. Present insights in whatever format best serves the research objective

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
[Your final analysis and reasoning about what format and depth best suits this research task]
</thinking>

<complete>
[Present your research findings in whatever structure and depth best addresses your assigned task. Let the nature of the topic guide your approach - some tasks need concise summaries, others require comprehensive analysis.

Always include proper citations [1][2] for all claims and data throughout your response.]

## References

1. **Title** - Date - [domain.com](https://full-url.com/path)
2. **Title** - Date - [domain.com](https://full-url.com/path)
3. **Title** - Date - [domain.com](https://full-url.com/path)
</complete>

CRITICAL REQUIREMENTS:
- Every data point, opinion, and fact MUST have citations [1][2] etc
- References MUST follow EXACT format: **Title** - Date - [domain.com](https://full-url)
- Each reference on its own numbered line under "References" section
- Let the research task complexity naturally determine your response depth and structure
- Format: number. **Title** - Date - [domain.com](https://complete-url)
- Extract real URLs from the sources you searched and read
- Include complete working URLs, not just domain names
- NEVER truncate titles with "..." - always write complete, descriptive titles
- Use markdown link format: [domain.com](https://complete-url)
- CRITICAL: Reference titles must be complete and descriptive, never truncated

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

MINIMIZE ITERATIONS: Aim to complete research in 1-2 thinking+action cycles maximum.

FINAL REFERENCE REMINDER: When writing your references section, you MUST:
- Use the EXACT URLs from the search results and read operations above
- Format: **Title** - Date - [domain.com](https://complete-url-from-search)
- Copy the full URLs from the "可用引用源" section provided after each search
- Do NOT create generic domain.com URLs - use real working links
- ABSOLUTELY CRITICAL: Write complete titles, never use "..." or truncate reference titles
- Example of CORRECT format: **2025年中国生猪养殖供给市场深度分析报告** - 2024-12-15 - [research.com](https://research.com/pig-market-2025)
- Example of WRONG format: **2025年中国生猪养殖供给市场分析...** - NEVER DO THIS"""

        super().__init__(
            role=f"researcher_{role}",
            system_prompt=system_prompt,
            llm_client=llm_client,
            model=model,
            temperature=0.7,
            memory_store=memory_store,
        )

        self.research_area = role
        self.initial_task = initial_task
        self.context.current_goal = initial_task

    def _parse_llm_output(self, output: str) -> AgentResponse:
        """
        Parse researcher's output to determine next action(s)
        """

        response = AgentResponse(thought=output)

        # Find all action tags
        import re

        action_pattern = r"<action>(.*?)</action>"
        action_matches = re.findall(action_pattern, output, re.DOTALL)

        if action_matches:
            actions = []
            for action_content in action_matches:
                # Parse each action content
                lines = action_content.strip().split("\n")
                tool_line = next((line for line in lines if line.startswith("TOOL:")), None)
                input_line = next((line for line in lines if line.startswith("INPUT:")), None)

                if tool_line and input_line:
                    tool = tool_line.split(":", 1)[1].strip().upper()
                    input_value = input_line.split(":", 1)[1].strip()

                    if tool == "SEARCH":
                        actions.append(
                            AgentAction(
                                action_type=ActionType.USE_TOOL, parameters={"tool": "search", "query": input_value}
                            )
                        )
                    elif tool == "READ":
                        actions.append(
                            AgentAction(
                                action_type=ActionType.USE_TOOL, parameters={"tool": "read", "url": input_value}
                            )
                        )

            # Store all actions (we'll modify AgentResponse to handle multiple actions)
            if len(actions) == 1:
                response.action = actions[0]
            elif len(actions) > 1:
                # Store multiple actions in parameters for batch execution
                response.action = AgentAction(action_type=ActionType.USE_TOOL, parameters={"batch_actions": actions})

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
                MessageRole.SYSTEM, "Continue your research. What else do you need to investigate?"
            )

    async def _execute_batch_actions(self, actions: list):
        """
        Execute multiple actions with concurrent support for READ operations
        """
        import asyncio
        import logging
        
        # Group actions by tool type
        search_actions = []
        read_actions = []
        
        for action in actions:
            tool = action.parameters.get("tool")
            if tool == "search":
                search_actions.append(action)
            elif tool == "read":
                read_actions.append(action)
        
        # Execute searches sequentially (to maintain order)
        for action in search_actions:
            await self._execute_search(action.parameters.get("query"))
        
        # Execute reads concurrently if multiple
        if len(read_actions) > 1:
            logging.info(f"[ResearcherAgent] Executing {len(read_actions)} READ operations concurrently")
            
            # Create tasks for concurrent execution
            read_tasks = []
            for action in read_actions:
                url = action.parameters.get("url")
                read_tasks.append(self._execute_read(url))
            
            # Execute all reads concurrently
            await asyncio.gather(*read_tasks)
            
            logging.info(f"[ResearcherAgent] Completed {len(read_actions)} concurrent READ operations")
        elif len(read_actions) == 1:
            # Single read, execute normally
            await self._execute_read(read_actions[0].parameters.get("url"))

    async def _execute_search(self, query: str):
        """
        Execute a search action
        """
        if "search" not in self.available_tools:
            # Fallback to simulation if tool not available
            self.context.add_message(MessageRole.SYSTEM, "Search tool not available. Please configure tools.")
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
            citation_manager.add_citation_tracking_to_context(self.context, results)

            # Enhanced citation info with full URLs
            enhanced_citation_info = "\n\n--- 可用引用源 (请在参考文献中使用完整URL) ---\n"
            for i, result in enumerate(results, 1):
                title = result.get("title", "Unknown Title")
                url = result.get("url", "")
                date = result.get("date", "")
                domain = url.split("/")[2] if "://" in url else "unknown"
                enhanced_citation_info += f"[{i}] **{title}** - {date} - [{domain}]({url})\n"

            self.context.add_message(MessageRole.SYSTEM, enhanced_citation_info)

            # Format results for LLM
            if results:
                formatted_results = f"Search results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    formatted_results += f"{i}. {result['title']}\n"
                    # Add date if available
                    if "date" in result and result["date"]:
                        formatted_results += f"   Date: {result['date']}\n"
                    formatted_results += f"   URL: {result['url']}\n"
                    formatted_results += f"   {result['snippet']}\n\n"
            else:
                formatted_results = f"No search results found for '{query}'"

            # Add search results to context

            self.context.add_message(MessageRole.SYSTEM, formatted_results)

            # Store as collected info
            info = InfoPiece(
                content=formatted_results,
                source=f"search: {query}",
                metadata={"type": "search_results", "query": query, "results": results},
            )
            self.context.add_info(info)

        except Exception as e:
            self.context.add_message(MessageRole.SYSTEM, f"Error performing search: {str(e)}")

    async def _execute_read(self, url: str):
        """
        Execute a read action
        """
        import logging
        logging.info(f"[ResearcherAgent] Starting READ operation for URL: {url}")
        
        if "read" not in self.available_tools:
            # Fallback to simulation if tool not available
            self.context.add_message(MessageRole.SYSTEM, "Read tool not available. Please configure tools.")
            return

        read_tool = self.available_tools["read"]

        try:
            # Record read start
            if self.memory_store:
                self.memory_store.add_read_operation(self.role, url, "reading")

            logging.info(f"[ResearcherAgent] Executing read_url for: {url}")
            # Execute actual read
            content = await read_tool.read_url(url)
            logging.info(f"[ResearcherAgent] Read completed for: {url}, content length: {len(content) if content else 0}")

            # Check if content is None
            if content is None:
                self.context.add_message(MessageRole.SYSTEM, f"Failed to read content from {url}")
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
                self.context.add_message(MessageRole.SYSTEM, f"Content from {url} (summarized):\n\n{summary}")

                # Store original content as collected info
                info = InfoPiece(
                    content=content, source=url, metadata={"type": "article_content", "url": url, "summarized": True}
                )
                self.context.add_info(info)
            else:
                # Record completed without summary
                if self.memory_store:
                    self.memory_store.add_read_operation(self.role, url, "completed", "", content)

                # Add short content directly
                self.context.add_message(MessageRole.SYSTEM, content)
                logging.info(f"[ResearcherAgent] Added content to context (no summary needed)")

                # Store as collected info
                info = InfoPiece(content=content, source=url, metadata={"type": "article_content", "url": url})
                self.context.add_info(info)

        except Exception as e:
            logging.error(f"[ResearcherAgent] Error reading from {url}: {str(e)}")
            self.context.add_message(MessageRole.SYSTEM, f"Error reading {url}: {str(e)}")
        
        logging.info(f"[ResearcherAgent] READ operation fully completed for URL: {url}")

    def _extract_key_findings(self, output: str) -> str:
        """
        Extract and format key findings from researcher's output
        """
        # Look for structured findings
        findings_start = re.search(r"(?:key findings|findings|conclusions?|summary):", output, re.IGNORECASE)

        if findings_start:
            findings = output[findings_start.end() :].strip()
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
                {"role": "user", "content": summarize_prompt},
            ]

            # Generate hash for cache comparison

            response = await self.llm_client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3  # Lower temperature for more focused summaries
            )

            summary = response.choices[0].message.content
            return summary

        except Exception as e:
            logging.error(f"Error summarizing content: {e}")
            # Fallback: truncate content
            return content[:1500] + "...\n[Content truncated due to length]"
