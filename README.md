# Deep Research Agentic System

##  Overview
This project implements an *agent-based research system* that performs structured deep web research with reasoning, synthesis, and report generation.  
It uses a team of specialized AI agents coordinated by an *Orchestrator Agent*.  

The workflow:  
1. *PlanningAgent* – breaks down the user query into research tasks  
2. *WebSearchAgent* – performs real searches using the Tavily API  
3. *ReflectionAgent* – checks if enough evidence has been gathered  
4. *SynthesisAgent* – organizes findings into structured insights  
5. *ReportWriterAgent* – produces a polished, cited research report  
6. *OrchestratorAgent* – coordinates the entire process  

---

##  Features
- Automated multi-step research with *planning, search, reflection, synthesis, and reporting*  
- Real web search using [Tavily API](https://tavily.com) (10 calls/min rate-limited)  
- Retry and backoff logic to handle *rate limits* and *quota errors*  
- Final output saved as a Markdown report (.md) with *citations*  

---

##  Setup

1. Clone the Project
```bash
git clone <your-repo-url>
cd <project-folder>


2. Create Virtual Environment
bash
Copy code
uv venv
source .venv/Scripts/activate    # Windows
# or
source .venv/bin/activate        # Mac/Linux


3. Install Dependencies
bash
Copy code
uv pip install -r requirements.txt
Alternatively, you can install packages directly without requirements.txt:

bash
Copy code
uv pip install python-dotenv tavily openai


4. Add Environment Variables
Create a .env file with your keys:

bash
Copy code
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here


RUNNING THE RESEARCH SYSTEM

Run the orchestrator with your query:

bash
Copy code
uv run main.py
By default, it executes a deep research task and saves the report as:

Copy code
lead_generation_business_consultancy_report.md
You can edit the query in main.py:

python
Copy code
result = safe_run_sync(
    orchestrator_agent,
    "Do deep search for a lead generation system for a Pakistan-based business services consultancyy.",
    max_turns=20
)

print(result.final_output)


FILE STRUCTURE

bash
Copy code
main.py             # Core script (agents + orchestration + execution)
README.md           # Documentation
requirements.txt    # Python dependencies (if used)
.env                # API keys


AGENTS

Agent	Purpose
OrchestratorAgent	Controls the full workflow
PlanningAgent	Creates structured research steps
WebSearchAgent	Searches the web with Tavily
ReflectionAgent	Decides if evidence is sufficient
SynthesisAgent	Merges raw findings into insights
ReportWriterAgent	Generates the final professional report



Example Research Prompt
python
Copy code
result = safe_run_sync(
    orchestrator_agent,
    "Compare digital marketing lead generation strategies in the US vs Pakistan for SMEs.",
    max_turns=20
)

# Save the report
from pathlib import Path
Path("lead_generation_report.md").write_text(result.final_output, encoding="utf-8")