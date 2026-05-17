import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research, get_history

# Init
init_db()

app = FastAPI(title="Deep Research Agent v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ResearchRequest(BaseModel):
    query: str
    user_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research")
async def research(payload: ResearchRequest):
    # Build initial message
    user_content = payload.query

    # Run the graph
    result = await deep_research_graph.ainvoke({
        "messages": [HumanMessage(content=user_content)],
        "user_id": payload.user_id,
        "supervisor_messages": [],
        "raw_notes": [],
        "notes": [],
        "final_report": "",
        "research_brief": "",
    })

    final_report = result.get("final_report", "No report generated.")

    # Save to SQLite
    research_id = save_research(payload.user_id, payload.query, {
        "research_brief": result.get("research_brief", ""),
        "final_report": final_report,
        "notes": result.get("notes", []),
    })

    return {
        "research_id": research_id,
        "query": payload.query,
        "final_report": final_report,
    }


@app.get("/history/{user_id}")
def history(user_id: str, limit: int = 10):
    return {"user_id": user_id, "history": get_history(user_id, limit)}


