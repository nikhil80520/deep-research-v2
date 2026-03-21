import os
import json
import re
from typing import Literal
from langchain_core.messages import AIMessage, get_buffer_string
from langgraph.types import Command

from src.graph.state import AgentState, ClarifyWithUser


CLARIFY_SYSTEM = """Analyze the user's research request. Determine if clarification is needed.

Ask ONE targeted question only if:
- The topic has ambiguous acronyms
- The scope is completely unclear
- Critical missing info (audience, depth, specific angle)

Do NOT ask if you can reasonably infer intent.

If you have already asked a clarifying question in message history, do NOT ask again.

Respond in valid JSON:
{
  "need_clarification": true/false,
  "question": "Your question (empty string if not needed)",
  "verification": "Acknowledgement message (empty string if asking question)"
}"""


def _get_client():
    from cerebras.cloud.sdk import Cerebras
    return Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))


def _parse_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"need_clarification": False, "question": "", "verification": "Starting research now."}


async def clarify_with_user(state: AgentState, config=None) -> Command[Literal["write_research_brief", "__end__"]]:
    """Decide whether to ask a clarifying question or proceed to research."""
    messages = state.get("messages", [])
    msg_str = get_buffer_string(messages) if messages else ""

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": CLARIFY_SYSTEM},
                {"role": "user", "content": f"Messages so far:\n{msg_str}\n\nShould I ask a clarifying question?"},
            ],
            max_tokens=300,
        )
        result = _parse_json(response.choices[0].message.content)
    except Exception:
        result = {"need_clarification": False, "question": "", "verification": "Starting research now."}

    if result.get("need_clarification") and result.get("question"):
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=result["question"])]}
        )
    else:
        verification = result.get("verification") or "Got it! Starting research now."
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=verification)]}
        )
