import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research
from src.memory.mem0_client import init_memory, add_memory


async def main():
    init_db()
    init_memory()

    query = input("Enter research query: ").strip()
    if not query:
        query = "What are the latest advances in quantum computing in 2025?"

    user_id = "cli_user"
    print(f"\nStarting deep research on: {query}\n{'='*60}\n")

    result = await deep_research_graph.ainvoke({
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "supervisor_messages": [],
        "raw_notes": [],
        "notes": [],
        "final_report": "",
        "research_brief": "",
    })

    report = result.get("final_report", "No report generated.")
    brief = result.get("research_brief", "")

    print(f"\nResearch Brief:\n{brief}\n")
    print(f"\n{'='*60}\nFINAL REPORT\n{'='*60}\n")
    print(report)

    # Save
    save_research(user_id, query, {
        "research_brief": brief,
        "final_report": report,
        "notes": result.get("notes", []),
    })
    add_memory(user_id, query, report[:300])
    print("\n✅ Research saved to SQLite and mem0.")


if __name__ == "__main__":
    asyncio.run(main())
