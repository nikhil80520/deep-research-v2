import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research, get_history
from src.memory.mem0_client import init_memory, add_memory, search_memory, get_all_memories

# Init
init_db()
init_memory()

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
    # Get relevant past memory for context
    memory_ctx = search_memory(payload.user_id, payload.query)

    # Build initial message
    user_content = payload.query
    if memory_ctx:
        user_content = f"{memory_ctx}\n\n---\n\nCurrent query: {payload.query}"

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

    # Save to mem0
    add_memory(payload.user_id, payload.query, final_report[:300])

    return {
        "research_id": research_id,
        "query": payload.query,
        "final_report": final_report,
        "memory_context_used": bool(memory_ctx),
    }


@app.get("/history/{user_id}")
def history(user_id: str, limit: int = 10):
    return {"user_id": user_id, "history": get_history(user_id, limit)}


@app.get("/memories/{user_id}")
def memories(user_id: str):
    return {"user_id": user_id, "memories": get_all_memories(user_id)}
