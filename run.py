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

    # Validate Cerebras API key before starting
    cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
    if not cerebras_key:
        print("❌ CEREBRAS_API_KEY is not set. Please add it to your .env file.")
        return
    try:
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=cerebras_key)
        client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception as e:
        print(f"❌ Cerebras API key validation failed: {e}")
        print("   Please check your CEREBRAS_API_KEY in the .env file.")
        return

    query = input("Enter research query: ").strip()
    if not query:
        query = "What are the latest advances in quantum computing in 2025?"

    user_id = "cli_user"
    messages = [HumanMessage(content=query)]

    # Clarification loop — re-invoke if the graph asks a clarifying question
    max_clarifications = 3
    for attempt in range(max_clarifications + 1):
        print(f"\nStarting deep research on: {query}\n{'='*60}\n")

        result = await deep_research_graph.ainvoke({
            "messages": messages,
            "user_id": user_id,
            "supervisor_messages": [],
            "raw_notes": [],
            "notes": [],
            "final_report": "",
            "research_brief": "",
        })

        report = result.get("final_report", "")
        if report:
            # Research completed successfully
            break

        # No report — check if the clarifier asked a question
        result_messages = result.get("messages", [])
        last_ai_msg = None
        for msg in reversed(result_messages):
            if hasattr(msg, "content") and msg.content:
                last_ai_msg = msg
                break

        if last_ai_msg and attempt < max_clarifications:
            print(f"🤔 Clarification needed: {last_ai_msg.content}")
            try:
                answer = input("\nYour answer: ").strip()
            except EOFError:
                print("No input available. Proceeding with original query.")
                break
            if answer:
                messages = result_messages + [HumanMessage(content=answer)]
                continue
        break

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
    add_memory(user_id, query, report[:300] if isinstance(report, str) else str(report)[:300])
    print("\n✅ Research saved to SQLite and mem0.")


if __name__ == "__main__":
    asyncio.run(main())

