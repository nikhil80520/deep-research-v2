import os
import json
import re
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string
from langgraph.types import Command

from src.graph.state import AgentState


BRIEF_SYSTEM = """Transform the user's research request into a detailed, structured research brief.

The brief should:
1. State the core research question clearly
2. List specific aspects to investigate
3. Mention desired depth/angle if apparent
4. Specify any constraints (recency, geography, etc.)
5. Be explicit — no acronyms, full context

Respond in JSON:
{
  "research_brief": "Detailed research brief..."
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
    # Fallback
    return {"research_brief": text.strip()}


async def write_research_brief(state: AgentState, config=None) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief."""
    messages = state.get("messages", [])
    msg_str = get_buffer_string(messages) if messages else "No messages."

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": BRIEF_SYSTEM},
                {"role": "user", "content": f"User's research request:\n{msg_str}\n\nCreate a detailed research brief."},
            ],
            max_tokens=600,
        )
        result = _parse_json(response.choices[0].message.content)
    except Exception as e:
        result = {"research_brief": msg_str}

    brief = result.get("research_brief", msg_str)

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    {"role": "system", "content": "You are a research supervisor."},
                    {"role": "user", "content": brief},
                ]
            }
        }
    )
