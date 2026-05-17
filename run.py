import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research


async def main():
    init_db()

    from src.config.configuration import Configuration

    # Validate AWS Bedrock credentials before starting
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id = Configuration.resolve_bedrock_model_id()

    if not aws_access_key_id or not aws_secret_access_key:
        print("❌ AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required in .env")
        return
    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
        )
        # Lightweight capability check; verifies credentials and model access.
        client.converse(
            modelId=bedrock_model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 1, "temperature": 0},
        )
    except Exception as e:
        print(f"❌ AWS Bedrock validation failed: {e}")
        print("   Check AWS credentials, region, and BEDROCK_MODEL_ID/BEDROCK_LLM_MODEL in .env")
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
    print("\n✅ Research saved to SQLite.")


if __name__ == "__main__":
    asyncio.run(main())

