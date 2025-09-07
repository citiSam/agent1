import os
import time
import pathlib
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from tavily import AsyncTavilyClient

# -------------------------------------------------------------------
# 1. Load environment variables
# -------------------------------------------------------------------
load_dotenv(find_dotenv())

# Gemini key for LLMs
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# Tavily key for search
# tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
 

# -------------------------------------------------------------------
# 2. LLM Clients
# -------------------------------------------------------------------
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

light_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite", openai_client=external_client
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=external_client
)

special_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-pro", openai_client=external_client
)

# -------------------------------------------------------------------
# 3. Web Search Tool (Tavily with Rate Limiting)
# -------------------------------------------------------------------

last_call_time = 0


@function_tool
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a real web search using Tavily API (rate-limited to 10 calls/min).
    Returns formatted results with title, URL, and snippet/content.
    """
    global last_call_time
    elapsed = time.time() - last_call_time
    if elapsed < 6:  # 10 calls/min limit => 1 call every 6s
        time.sleep(6 - elapsed)

    last_call_time = time.time()

    try:
        results = await tavily_client.search(query=query, max_results=max_results)
        print("DEBUG Tavily response:", results)  #  log raw response

        if "results" not in results or not results["results"]:
            return "No results found."

        formatted = []
        for r in results["results"]:
            title = r.get("title", "No Title")
            url = r.get("url", "No URL")
            snippet = r.get("snippet") or r.get("content") or "No snippet available"
            formatted.append(f"- {title} ({url}): {snippet}")

        return "\n".join(formatted)

    except Exception as e:
        return f"WebSearchTool failed: {str(e)}"


# -------------------------------------------------------------------
# 4. Utility Tool
# -------------------------------------------------------------------
@function_tool
def current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# -------------------------------------------------------------------
# 5. Agents
# -------------------------------------------------------------------
websearch_agent: Agent = Agent(
    name="WebSearchAgent",
    instructions="You are a helpful web search assistant. Use web search to find relevant information and return results with citations.",
    model=llm_model,
    tools=[web_search],
)

reflection_agent: Agent = Agent(
    name="ReflectionAgent",
    instructions="You are a helpful reflection assistant. Use reflective listening to decide if the research goal is complete.",
    model=light_model,
    tools=[],
)

planning_agent: Agent = Agent(
    name="PlanningAgent",
    instructions="You are a helpful planning assistant. Plan the research process step by step using scientific methods. Always explain the principles you used.",
    model=special_model,
    tools=[],
)

synthesis_agent = Agent(
    name="synthesis_agent",
    instructions="""
    Your job is to review all research notes and sources,
    merge overlapping findings, resolve contradictions,
    and create a clear, structured summary.
    - Group insights into categories
    - Remove duplicates
    - Flag uncertainties
    - Keep track of citations
    Return a clean knowledge base for the report writer.
    """,
    model=light_model,
    tools=[],
)

report_writer_agent = Agent(
    name="report_writer_agent",
    instructions="""
    You are the Report Writer Agent.
    Using the structured insights from the synthesis_agent,
    produce a professional research report.
    - Add an executive summary
    - Organize with clear sections & subheadings
    - Insert citations inline (with URLs)
    - End with a conclusion and a references section
    Write in a professional, academic style, suitable for clients or publication.
    """,
    model=light_model,
    tools=[],
)

orchestrator_agent: Agent = Agent(
    name="OrchestratorAgent",
    instructions="""
       You are the Orchestrator Agent.
    Process for each deep research request:

    1. Get current date
    2. Ask the planning_agent to create a research plan
    3. Delegate web search tasks to websearch_agent
    4. Use reflection_agent to verify sufficiency & credibility of results
    5. Pass collected findings to synthesis_agent to merge & organize
    6. Finally, pass synthesized insights to report_writer_agent to draft the final report

    Always ensure the final output includes citations and is well-structured.
    Stop the workflow once the report is written.

Stopping rules:
- When you have gathered enough evidence to answer the request, summarize and END the task immediately.
- Do not continue asking the search agent once you can provide a reasonable, well-cited answer.
- Never exceed 7 total exchanges with other agents. If the goal is not fully achieved by then, provide the best summary you can and stop.

    """,
    model=llm_model,
    tools=[
        current_date,
        planning_agent.as_tool("PlanningTool", "Planning assistant with scientific reasoning"),
        reflection_agent.as_tool("ReflectionTool", "Reflection assistant"),
        websearch_agent.as_tool("WebSearchTool", "Real web search assistant with citations"),
        report_writer_agent.as_tool("ReportWriterTool", "Creates the final polished research report with citations"),
        synthesis_agent.as_tool("SynthesisTool", "Merges and organizes research findings"),
    ],
)


# -------------------------------------------------------------------
# 6. Run Deep Research
# -------------------------------------------------------------------

# Throttled wrapper for Runner.run_sync
def throttled_run_sync(agent, prompt):
    time.sleep(7)  # pause ~7s before each call
    return Runner.run_sync(agent, prompt)


#  Retry wrapper with backoff
import time
from openai import RateLimitError


def safe_run_sync(agent, prompt, max_turns=10, max_retries=5, base_delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            return Runner.run_sync(agent, prompt, max_turns=max_turns)
        except RateLimitError as e:
            wait_time = base_delay * (2 ** attempt)  # exponential backoff
            print(f"Rate limit hit (attempt {attempt+1}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = base_delay * (2 ** attempt)
                print(f"Gemini quota exceeded (attempt {attempt+1}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
            else:
                raise
    raise Exception("Too many retries, giving up.")



# Now call it here
result = safe_run_sync(
    orchestrator_agent,
    "Do deep search for a lead generation system for a professiona business services consultancy.",
    max_turns=20
)

print(result.final_output)

# result = safe_run_sync(...)  # or however you call Runner now
report_text = result.final_output

# Save to disk
from pathlib import Path
out_path = Path("lead_generation_business_consultancy_report.md").resolve()
out_path.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to: {out_path}\n")

# Optionally print only the report (no debug)
print(report_text)