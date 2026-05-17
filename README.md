# 🔬 Deep Research Agent

A multi-agent, deep research system powered by LangGraph, AWS Bedrock, and Tavily. The system uses a supervisor-researcher pattern where a supervisor agent delegates parallel research tasks to specialized sub-researchers, each conducting autonomous web searches and synthesizing findings to produce a comprehensive final report.

![Deep Research Dashboard](https://img.shields.io/badge/UI-Streamlit_Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-000000?style=flat-square)
![AWS Bedrock](https://img.shields.io/badge/LLM-AWS_Bedrock-232F3E?style=flat-square&logo=amazon-aws)

<video src="demo.mp4" controls="controls" muted="muted" width="800"></video>

## ✨ Key Features

*   **Multi-Agent Architecture**: Built with LangGraph. A Supervisor agent plans the research and delegates to multiple Researcher agents running in parallel.
*   **Interactive Real-Time Dashboard**: A high-fidelity Streamlit UI featuring:
    *   **Command Center**: Live status badge and a pipeline progress breadcrumb (Intent Analysis → Parallel Extraction → Synthesis → Final Report).
    *   **Active Intelligence Feed**: Real-time action cards detailing agent steps (Thinking blocks, Web Searches, Tool Invocations, and Source Chips with domain favicons).
    *   **Knowledge Graph Sidebar**: Live metrics tracking active researchers, search rounds, scanned sites, and data volume fetched.
*   **Autonomous Web Research**: Agents utilize the Tavily Search API to gather real-time data, avoiding hallucinations and grounding the final report in facts.
*   **Intelligent Reasoning & Reflection**: Agents use specialized `think_tool` blocks to pause, assess their findings, and plan their next searches strategically.
*   **Robust Memory & Persistence**: Research history is stored in an SQLite database, allowing users to review past comprehensive reports.

## 🏗️ System Architecture

The research pipeline operates as a state machine:

1.  **Intent Analysis (`clarifier.py`)**: Analyzes if the user's query needs clarification. If ambiguous, it asks a targeted question.
2.  **Brief Writing (`brief_writer.py`)**: Transforms the verified user input into a structured research brief.
3.  **Research Supervisor (`supervisor.py`)**: A subgraph that orchestrates the research.
    *   Uses an AWS Bedrock LLM with tool calling to decide the research strategy.
    *   Delegates topics to up to 5 parallel researchers via `asyncio.gather()`.
4.  **Parallel Researchers (`researcher.py`)**: Each researcher runs a ReAct loop to search the web, reflect on findings, and compress the gathered data into cited summaries.
5.  **Synthesis & Final Report (`report_writer.py`)**: Consolidates all individual researcher findings into a cohesive, comprehensive markdown report.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   [AWS Credentials for Bedrock](https://aws.amazon.com/bedrock/)
*   [Tavily API Key](https://tavily.com/)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/deep-research-v2.git
    cd deep-research-v2
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Copy the example `.env` file and add your API keys:
    ```bash
    cp .env.example .env
    ```
    *Inside `.env`:*
    ```env
    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=your_aws_access_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret_key
    BEDROCK_LLM_MODEL=qwen.qwen3-32b-v1:0
    TAVILY_API_KEY=your_tavily_api_key
    DB_PATH=research_history.db
    ```

## 💻 Usage

The application provides three different entry points depending on your needs.

### 1. Interactive Streamlit Dashboard (Recommended)
Launch the rich, real-time UI to watch the multi-agent system work visually:
```bash
streamlit run streamlit_app.py
```
*Access the dashboard at `http://localhost:8501`.*

### 2. Command Line Interface (CLI)
Run a research query directly from your terminal:
```bash
python run.py
```

### 3. FastAPI Server
Run the system as a headless API service:
```bash
uvicorn src.api.main:app --reload --port 8000
```
*   `POST /research` - Submit a new research query.
*   `GET /history/{user_id}` - Retrieve past research reports.

## 📂 Project Structure

```text
deep-research-v2/
├── .env                # Environment configuration
├── requirements.txt    # Python dependencies
├── run.py              # CLI entry point
├── streamlit_app.py    # Streamlit dashboard entry point
└── src/
    ├── api/            # FastAPI routes and server
    ├── agents/         # LangGraph agent definitions (supervisor, researcher, etc.)
    ├── config/         # System configuration and limits
    ├── graph/          # LangGraph state management and workflow routing
    ├── memory/         # SQLite database operations
    ├── tools/          # Agent tools (Tavily search, think tool)
    └── ui/             # Streamlit UI components (Renderer, Trace events, Styles)
```

## 🛠️ Built With
*   [LangGraph](https://python.langchain.com/docs/langgraph) - For building stateful, multi-actor applications with LLMs.
*   [Streamlit](https://streamlit.io/) - For the interactive frontend dashboard.
*   [AWS Bedrock](https://aws.amazon.com/bedrock/) - For scalable, enterprise-grade LLM inference.
*   [Tavily](https://tavily.com/) - Search API optimized for AI agents.

## 📄 License
This project is licensed under the MIT License.
