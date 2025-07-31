"""
Supervisor Agent that decomposes research tasks and coordinates multiple researcher agents
"""

import re
import uuid
from typing import List, Dict, Any
from datetime import datetime

from .core import LLMAgent, AgentResponse, AgentAction, ActionType, InfoPiece, MessageRole
from .researcher import ResearcherAgent


class SupervisorAgent(LLMAgent):
    """
    Supervisor agent that manages the overall research process
    """

    def __init__(self, llm_client=None, model="gpt-4-turbo-preview", tools=None, memory_store=None):
        current_date = datetime.now().strftime("%Y-%m-%d")

        system_prompt = f"""You are a Research Supervisor AI that manages complex research projects.

Current Date: {current_date}

CRITICAL LANGUAGE REQUIREMENT: You MUST respond in the EXACT SAME LANGUAGE as the user's query throughout the entire research process.
- If user query is in Chinese → ALL responses, analysis, and final report MUST be in Chinese
- If user query is in English → ALL responses, analysis, and final report MUST be in English
- This rule applies to ALL outputs including thinking, actions, and final synthesis
- NEVER switch languages unless explicitly requested by the user

CRITICAL ROLE CLARIFICATION:
- Researchers are your INFORMATION GATHERING TOOLS, not analysts or writers
- Researchers ONLY collect raw data, facts, and basic summaries - nothing more
- YOU are the senior analyst who creates ALL final conclusions, analysis, and reports
- NEVER delegate analysis, synthesis, or report writing to researchers
- Use researchers to gather information, then YOU do all the thinking and comprehensive writing

Your responsibilities:
1. Understand what information you need to answer the research question
2. Create researchers ONLY to COLLECT specific information (facts, data, sources)
3. When you have sufficient raw information, YOU write the final comprehensive analysis
4. Do NOT create "integration specialists", "analysis experts", or "report writers" - that's YOUR job
5. Provide final answers that match the natural scope and depth of the original question

WORKFLOW PRINCIPLE:
- Researchers = Information collectors
- Supervisor (YOU) = Analyst, synthesizer, and report writer

IMPORTANT: 
- Create only ONE researcher per step. Do not create multiple researchers simultaneously.
- When researchers complete their work, they provide reports with citations [1][2] and references
- You MUST preserve and use these citations in your final report
- A professional research report always includes citations and references

You have access to the following tool:
- CREATE_RESEARCHER: Create a specialist researcher for a specific task

When you want to create a researcher, use this format:
<action>
TOOL: CREATE_RESEARCHER
INPUT: role_name|specific_research_task
</action>

When you have collected sufficient information from researchers, YOU must provide the final comprehensive analysis:
<complete>
[THIS IS YOUR CORE RESPONSIBILITY: Write the complete, final analysis based on all collected information. 
- The format, depth, and structure should naturally match what the original query requires
- Synthesize all researcher findings into YOUR comprehensive response
- Include proper citations [1][2] from the researchers' reports
- Add a complete references section at the end
- This is YOUR analysis and conclusions, not a summary of researcher reports]
</complete>

CRITICAL: The final report is YOUR work as the senior analyst. Researchers only provided raw materials - YOU create the final comprehensive response that directly answers the user's question.

Work step by step - create one researcher, wait for results, then decide on next steps.
"""

        super().__init__(
            role="supervisor",
            system_prompt=system_prompt,
            llm_client=llm_client,
            model=model,
            memory_store=memory_store,
        )

        self.researchers: Dict[str, ResearcherAgent] = {}
        self.research_tasks: List[Dict[str, Any]] = []

        # Use provided tools or create new ones
        if tools:
            self.tools = tools
        else:
            from .tools import create_tools

            self.tools = create_tools()

    def _parse_llm_output(self, output: str) -> AgentResponse:
        """
        Parse supervisor's output using same XML format as researcher
        """
        import logging

        logging.debug(f"[Supervisor] LLM Output: {output[:500]}...")
        response = AgentResponse(thought=output)

        # Check for action tag (same as researcher)
        if "<action>" in output and "</action>" in output:
            action_start = output.find("<action>") + 8
            action_end = output.find("</action>")
            action_content = output[action_start:action_end].strip()

            # Parse action content
            lines = action_content.split("\n")
            tool_line = next((line for line in lines if line.startswith("TOOL:")), None)
            input_line = next((line for line in lines if line.startswith("INPUT:")), None)

            if tool_line and input_line:
                tool = tool_line.split(":", 1)[1].strip().upper()
                input_value = input_line.split(":", 1)[1].strip()

                if tool == "CREATE_RESEARCHER":
                    # Parse role and task from input: "role_name|task_description"
                    if "|" in input_value:
                        role, task = input_value.split("|", 1)
                        response.needs_specialist = True
                        response.specialist_role = role.strip()
                        response.sub_task = task.strip()
                        response.action = AgentAction(
                            action_type=ActionType.CREATE_AGENT, parameters={"role": role.strip(), "task": task.strip()}
                        )

        # Check for completion tag
        elif "<complete>" in output and "</complete>" in output:
            complete_start = output.find("<complete>") + 10
            complete_end = output.find("</complete>")
            final_report = output[complete_start:complete_end].strip()

            response.is_complete = True
            response.result = final_report

        else:
            # Continue thinking if no structured output
            response.action = AgentAction(action_type=ActionType.CONTINUE_THINKING)

        return response

    async def _execute_action(self, action: AgentAction):
        """
        Execute the action determined by LLM
        """
        if action.action_type == ActionType.CREATE_AGENT:
            await self._create_researcher(action.parameters["role"], action.parameters["task"])
        elif action.action_type == ActionType.CONTINUE_THINKING:
            # Add a system message to prompt continued analysis
            self.context.add_message(MessageRole.SYSTEM, "Continue your analysis. What else do you need to know?")

    async def _create_researcher(self, role: str, task: str):
        """
        Create a new researcher agent for a specific task
        """
        import logging

        researcher_id = f"{role.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

        logging.info(f"[CREATE_RESEARCHER] Role: {role}, Task: {task}")
        logging.info(f"[CREATE_RESEARCHER] Researcher ID: {researcher_id}")

        # Create researcher with specific role and task
        researcher = ResearcherAgent(
            role=role, initial_task=task, llm_client=self.llm_client, model=self.model, memory_store=self.memory_store
        )

        # Register tools with researcher
        for name, tool in self.tools.items():
            researcher.register_tool(name, tool)

        self.researchers[researcher_id] = researcher
        self.research_tasks.append(
            {
                "researcher_id": researcher_id,
                "role": role,
                "task": task,
                "status": "in_progress",
                "created_at": datetime.now(),
            }
        )

        # Let researcher complete their research
        self.context.add_message(MessageRole.SYSTEM, f"Creating {role} to research: {task}")

        logging.info(f"[RESEARCHER_START] Starting research for {role}")

        # Run researcher until complete
        result = await researcher.run_until_complete()

        logging.info(f"[RESEARCHER_COMPLETE] {role} finished, result length: {len(result)}")

        # 详细记录 researcher 的输出，特别是引用部分
        logging.debug(f"[RESEARCHER_OUTPUT] {role} full result:\n{result}")

        # 查找引用部分
        ref_pos = result.find("References")
        if ref_pos == -1:
            ref_pos = result.find("参考文献")

        if ref_pos != -1:
            logging.info(f"[RESEARCHER_OUTPUT] {role} has references at position {ref_pos}")
            # 提取引用部分到结尾
            ref_section = result[ref_pos:]
            logging.info(f"[RESEARCHER_OUTPUT] {role} references section:\n{repr(ref_section[:500])}...")
            logging.info(f"[RESEARCHER_OUTPUT] {role} references (readable):\n{ref_section}")
        else:
            logging.warning(f"[RESEARCHER_OUTPUT] {role} has NO references section!")

        # 详细分析 researcher 的参考文献部分
        if "References" in result or "参考文献" in result:
            # 找到参考文献部分的开始
            ref_keywords = ["References", "参考文献", "## References", "## 参考文献"]
            ref_start_pos = -1
            ref_keyword_used = ""

            for keyword in ref_keywords:
                pos = result.find(keyword)
                if pos != -1:
                    ref_start_pos = pos
                    ref_keyword_used = keyword
                    break

            if ref_start_pos != -1:
                # 提取参考文献部分
                ref_section = result[ref_start_pos:]
                logging.info(
                    f"[REF_ANALYSIS] {role} - Reference keyword found: '{ref_keyword_used}' at position {ref_start_pos}"
                )
                logging.info(f"[REF_ANALYSIS] {role} - Reference section length: {len(ref_section)} characters")
                logging.info(f"[REF_ANALYSIS] {role} - Reference section preview: {repr(ref_section[:300])}")

                # 计算参考文献条目数量
                ref_entries = [line for line in ref_section.split("\n") if re.match(r"^\d+\.", line.strip())]
                logging.info(f"[REF_ANALYSIS] {role} - Found {len(ref_entries)} reference entries")

                # 检查是否有截断的迹象
                if ref_section.endswith("...") or ref_section.count("...") > 0:
                    logging.warning(f"[REF_ANALYSIS] {role} - TRUNCATION DETECTED in references: contains '...'")
                    truncated_lines = [line for line in ref_section.split("\n") if "..." in line]
                    for i, line in enumerate(truncated_lines):
                        logging.warning(f"[REF_ANALYSIS] {role} - Truncated line {i+1}: {repr(line)}")

                # 检查参考文献格式
                for i, entry in enumerate(ref_entries[:3]):  # 只检查前3个
                    logging.info(f"[REF_ANALYSIS] {role} - Entry {i+1}: {repr(entry.strip())}")
            else:
                logging.warning(f"[REF_ANALYSIS] {role} - No references section found despite indication")
        else:
            logging.warning(f"[REF_ANALYSIS] {role} - No references section found in result")

        # Add researcher's findings to supervisor's context
        findings = InfoPiece(
            content=result, source=f"{role} researcher", metadata={"researcher_id": researcher_id, "task": task}
        )
        self.context.add_info(findings)

        # Update task status
        for task_info in self.research_tasks:
            if task_info["researcher_id"] == researcher_id:
                task_info["status"] = "completed"
                task_info["completed_at"] = datetime.now()
                task_info["result"] = result

        # Inform supervisor of completion with full results including citations
        self.context.add_message(
            MessageRole.SYSTEM, f"{role} has completed their research on: {task}\n\nFull findings:\n{result}"
        )

        logging.info(f"[RESEARCHER_ADDED] Added {role} result to supervisor context")

        return result

    async def research(self, query: str) -> str:
        """
        Main entry point to conduct research on a topic
        """
        import logging

        logging.info(f"[SUPERVISOR_START] Query: {query}")
        logging.info(f"[SUPERVISOR_START] Initial context messages: {len(self.context.conversation_history)}")

        self.context.current_goal = query
        self.context.add_message(MessageRole.USER, query)

        # Run until complete (limit to 20 steps for debugging)
        result = await self.run_until_complete(max_steps=20)

        logging.info(f"[SUPERVISOR_COMPLETE] Run complete result length: {len(result) if result else 0}")
        logging.info(f"[SUPERVISOR_COMPLETE] Research tasks count: {len(self.research_tasks)}")

        # Generate final report
        if not result:
            logging.info("[SUPERVISOR_FINAL] No result from run_until_complete, generating final report")
            result = await self._generate_final_report()
        else:
            logging.info("[SUPERVISOR_FINAL] Using result from run_until_complete")

        return result

    async def _generate_final_report(self) -> str:
        """
        Generate comprehensive final report based on all research
        """
        report_prompt = f"""Based on all the research conducted, generate a comprehensive and detailed report.

Original Query: {self.context.current_goal}

CRITICAL LANGUAGE REQUIREMENT: Your final report MUST be written in the EXACT SAME LANGUAGE as the original query above. 
- Analyze the query language and respond accordingly
- If query is in Chinese, write the entire report in Chinese
- If query is in English, write the entire report in English
- Maintain language consistency throughout the entire report

IMPORTANT: Extract and use ALL citations from the research results below. Each researcher has already provided numbered citations - use them!

Research Tasks Completed:
"""

        # Collect all research results with their citations
        import logging

        for task in self.research_tasks:
            if task["status"] == "completed":
                report_prompt += f"\n- {task['role']}: {task['task']}"
                result = task.get("result", "No result")
                logging.debug(f"[Supervisor] Task {task['role']} result length: {len(result)}")
                logging.debug(f"[Supervisor] Result preview: {result[:200]}...")
                report_prompt += f"\n  Result: {result}\n"

        report_prompt += "\n\nCRITICAL URL EXTRACTION NOTICE: The above research results contain complete URLs in their References sections. You MUST:\n1. Find each researcher's References section\n2. Copy the EXACT URLs including the full https:// links\n3. Use these EXACT URLs in your final References section\n4. Do NOT create new URLs or omit the URL links\n\nExample: If researcher shows '1. **Title** - 2024-01-01 - [site.com](https://site.com/full/path)', you MUST copy https://site.com/full/path exactly."

        report_prompt += """
Please synthesize all findings into a comprehensive, professional research report with deep analysis and insights. Create a report that thoroughly covers all aspects of the research question.

## Report Structure Requirements:

1. **Executive Summary** - Comprehensive overview of research background, core findings and main conclusions
2. **Current State Analysis** - Detailed analysis of current situation with specific data and market conditions
3. **In-depth Analysis** - Multi-dimensional deep analysis based on research results, including:
   - Data analysis and trend identification
   - Key influencing factors analysis
   - Comparative analysis and benchmarking
4. **Key Findings** - Most important research discoveries and insights
5. **Opportunities & Challenges** - Identified opportunities and potential risk analysis  
6. **Trend Forecast** - Future development trend analysis based on data
7. **Conclusions & Recommendations** - Conclusions and actionable recommendations based on analysis

## Writing Requirements:
- Fully utilize all research findings to ensure comprehensive and substantial report content
- Every point must be supported by specific data and facts with citations [1][2] etc
- Provide deep analysis not simple listing, offer insights and judgments
- Maintain professionalism and logic with clear structure
- Use markdown format to organize content for better readability
- **MUST INCLUDE CITATIONS**: All data, opinions, and facts must have corresponding citation support

## Citation Requirements:
**CRITICAL**: The report MUST include citations and references section - this is a fundamental requirement for professional research reports.

1. **In-text Citations**: When citing data or opinions, add citation numbers after sentences, e.g.: Pork production reached 57.94 million tons[1]
2. **References Section**: The report MUST end with a complete reference list in this format:

References

1. **Title** - Date - [domain.com](https://domain.com)
2. **Title** - Date - [domain.com](https://domain.com)
3. **Title** - Date - [domain.com](https://domain.com)

CRITICAL FORMATTING REQUIREMENTS for references:
- Start with "References" header  
- Each reference numbered and on its own line
- Format: number. **Title** - Date - [domain.com](https://full-url)
- MANDATORY: Extract complete URLs from researcher reports - do NOT create new URLs
- COPY EXACT URLs from researcher reports: [domain.com](https://complete-url-from-researcher)
- Keep titles descriptive and accurate - do NOT truncate with "..."
- NEVER omit the URL link part - every reference MUST have clickable URL

Extract ALL important citation sources from the above research results to ensure report authority and credibility. A report without proper formatted citations is incomplete and unprofessional.

REMINDER: The researcher reports above already contain numbered citations and references. You MUST:
1. Use the same citation numbers when referring to the same sources
2. Compile all references from all researchers into a unified reference list at the end
3. Ensure every important data point has a citation

CRITICAL: YOU MUST FOLLOW THE EXACT REFERENCE FORMAT BELOW:

References

1. **Article Title** - 2024-01-15 - [example.com](https://example.com/full-url)
2. **Another Article** - 2024-02-20 - [site.org](https://site.org/article)

FORMATTING RULES (MANDATORY):
- References section MUST start with "References" header
- Each reference MUST be numbered: "1. ", "2. ", etc. (NOT [1], [2])
- Title MUST be in bold: **Title**
- Include real dates from researcher reports
- CRITICAL: Include clickable markdown links: [domain.com](https://full-url)
- EXTRACT EXACT URLs from researcher reports above - do NOT make up URLs
- EXAMPLE: If researcher report shows "[baijiahao.baidu.com](https://baijiahao.baidu.com/s?id=123)", you MUST copy this exact URL

FINAL CHECKS before submitting your report:
1. LANGUAGE CHECK: Verify that the ENTIRE report is written in the same language as the original query
2. CITATION FORMAT CHECK: Ensure all references follow the EXACT format above
3. URL CHECK: All reference URLs must be real, working links extracted from researcher reports
4. NUMBERING CHECK: References must use "1. " format, NOT "[1]" format

These checks are absolutely critical for user experience and functionality."""

        self.context.add_message(MessageRole.SYSTEM, report_prompt)

        # Get final synthesis from LLM
        final_response = await self.step()

        final_report = final_response.thought

        # 分析最终报告的参考文献部分
        logging.info(f"[FINAL_REPORT] Report length: {len(final_report)} characters")

        if "References" in final_report or "参考文献" in final_report:
            # 找到参考文献部分
            ref_keywords = ["References", "参考文献", "## References", "## 参考文献"]
            ref_start_pos = -1
            ref_keyword_used = ""

            for keyword in ref_keywords:
                pos = final_report.find(keyword)
                if pos != -1:
                    ref_start_pos = pos
                    ref_keyword_used = keyword
                    break

            if ref_start_pos != -1:
                ref_section = final_report[ref_start_pos:]
                logging.info(
                    f"[FINAL_REPORT] Reference keyword found: '{ref_keyword_used}' at position {ref_start_pos}"
                )
                logging.info(f"[FINAL_REPORT] Reference section length: {len(ref_section)} characters")
                logging.info(f"[FINAL_REPORT] Reference section preview: {repr(ref_section[:500])}")

                # 检查截断
                if ref_section.endswith("...") or ref_section.count("...") > 0:
                    logging.error("[FINAL_REPORT] TRUNCATION DETECTED in final report references!")
                    truncated_lines = [line for line in ref_section.split("\n") if "..." in line]
                    for i, line in enumerate(truncated_lines):
                        logging.error(f"[FINAL_REPORT] Truncated line {i+1}: {repr(line)}")

                # 统计参考文献条目
                ref_entries = [line for line in ref_section.split("\n") if re.match(r"^\d+\.", line.strip())]
                logging.info(f"[FINAL_REPORT] Found {len(ref_entries)} reference entries in final report")

                # 显示前几个条目
                for i, entry in enumerate(ref_entries[:3]):
                    logging.info(f"[FINAL_REPORT] Entry {i+1}: {repr(entry.strip())}")
            else:
                logging.warning("[FINAL_REPORT] No references section found despite indication")
        else:
            logging.warning("[FINAL_REPORT] No references section found in final report")

        return final_report

    def get_research_summary(self) -> Dict[str, Any]:
        """
        Get summary of all research activities
        """
        return {
            "total_researchers": len(self.researchers),
            "tasks": self.research_tasks,
            "collected_info": len(self.context.collected_info),
            "status": self.state.value,
        }
