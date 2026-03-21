# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent deep research system built with LangGraph. Uses a supervisor-researcher pattern where a supervisor agent delegates parallel research tasks to sub-researchers, each conducting web searches and synthesizing findings. The system integrates with Cerebras (LLM), Tavily (search), mem0 (memory), and SQLite (persistence).

## Running the Application

Three entry points are available:

```bash
# CLI mode
python run.py

# Streamlit web UI
streamlit run streamlit_app.py

# FastAPI server
uvicorn src.api.main:app --reload --port 8000
```

## Setup

```bash
# Virtual environment already exists at myenv/
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Environment variables (copy .env.example to .env)
cp .env.example .env
# Required: CEREBRAS_API_KEY, TAVILY_API_KEY, MEM0_API_KEY, DB_PATH
```

## Architecture

### Graph Workflow (src/graph/workflow.py)

The research pipeline is a LangGraph state machine:

1. **clarify_with_user** (src/agents/clarifier.py) - Analyzes if the query needs clarification; asks one targeted question if ambiguous
2. **write_research_brief** (src/agents/brief_writer.py) - Transforms user input into structured research brief
3. **research_supervisor** (src/agents/supervisor.py) - Subgraph that orchestrates parallel research
   - `supervisor_node`: Uses Cerebras with tool calling to decide research strategy
   - `supervisor_tools_node`: Executes tools; runs up to 5 researchers in parallel via `asyncio.gather()`
4. **final_report_generation** (src/agents/report_writer.py) - Synthesizes all findings into markdown report

### Researcher Subgraph (src/agents/researcher.py)

Each researcher runs a ReAct loop with:
- `tavily_search` tool for web search (returns top 5 results, deduplicated by URL)
- `think_tool` for strategic reflection (must be called alone, not in parallel)
- `finish_research` tool to signal completion

After the loop, `compress_research` (src/agents/compressor.py) consolidates raw findings into structured output with inline citations and sources.

### State Management (src/graph/state.py)

Uses TypedDict state classes with reducers:
- `AgentState`: Top-level state with messages, research_brief, notes, final_report
- `SupervisorState`: For supervisor subgraph with iteration limits
- `ResearcherState`: Individual researcher state with tool call tracking
- `override_reducer` (src/graph/reducers.py): Supports `{"type": "override", "value": X}` for full replacement or list append

### Tools (src/tools/)

- `search.py`: Tavily async search with URL deduplication
- `think.py`: Reflection tool for planning/assessment
- `research_tools.py`: Placeholder tool definitions for supervisor delegation

### Memory (src/memory/)

- `database.py`: SQLite persistence for research history
- `mem0_client.py`: Semantic memory integration (optional, gracefully degrades if not configured)

### Configuration (src/config/configuration.py)

Key limits (defined in `Configuration`):
- `max_concurrent_researchers`: 5
- `max_supervisor_iterations`: 6
- `max_researcher_tool_calls`: 10
- `max_search_results`: 5

## API Endpoints (src/api/main.py)

- `POST /research` - Run research query (accepts `query`, `user_id`)
- `GET /history/{user_id}` - Get research history from SQLite
- `GET /memories/{user_id}` - Get mem0 memories for user
- `GET /health` - Health check

## Important Patterns

1. **Tool Calling Constraints**: The `think_tool` must be called alone (never in parallel with search). Researcher agents enforce this by checking tool call batches.

2. **Token Management**: Both `compress_research` and `final_report_generation` implement retry logic with progressive character truncation (starts at 6000/8000 chars, reduces by 30% on failure).

3. **Parallel Research**: Supervisor runs multiple researchers concurrently via `asyncio.gather()` in `supervisor_tools_node`, then aggregates results back to supervisor state.

4. **LLM Integration**: All agents use Cerebras SDK directly with `llama3.1-8b` model and manual message formatting (not LangChain's chat model interface).

5. **State Commands**: Nodes return `Command[goto=..., update=...]` for conditional routing rather than hardcoded edges.
