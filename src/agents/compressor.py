import os
import re
from langchain_core.messages import filter_messages, HumanMessage, AIMessage, ToolMessage

from src.graph.state import ResearcherState


COMPRESS_SYSTEM = """You are a research assistant. You have conducted research by calling several search tools.
Your job is to clean up and consolidate the findings — but preserve ALL relevant information.

Rules:
1. Preserve ALL key facts, statistics, quotes, and data verbatim.
2. Remove only obviously duplicate or irrelevant information.
3. Add inline citations [1], [2], etc. for each source.
4. End with a ### Sources section listing all URLs.
5. Output can be as long as needed — do NOT summarize away details.
6. Format in clean markdown.

Output format:
## Research Findings

[Your consolidated findings with inline citations]

### Sources
1. [Title]: URL
2. [Title]: URL
"""


def _get_cerebras_client():
    from cerebras.cloud.sdk import Cerebras
    return Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))


def _extract_text(messages: list) -> str:
    """Extract all text content from researcher messages."""
    parts = []
    for msg in messages:
        content = ""
        if isinstance(msg, (AIMessage,)):
            content = str(msg.content)
        elif isinstance(msg, ToolMessage):
            content = str(msg.content)
        elif isinstance(msg, HumanMessage):
            content = str(msg.content)
        elif isinstance(msg, dict):
            content = str(msg.get("content", ""))
        if content.strip():
            parts.append(content)
    return "\n\n".join(parts)


async def compress_research(state: ResearcherState) -> dict:
    """Compress raw researcher messages into clean findings with citations.
    
    Handles token limits by progressively truncating input and retrying.
    """
    researcher_messages = state.get("researcher_messages", [])
    raw_text = _extract_text(researcher_messages)

    client = _get_cerebras_client()
    max_attempts = 3
    char_limit = 6000  # Start with 6k chars of raw text

    for attempt in range(max_attempts):
        try:
            truncated = raw_text[:char_limit]
            response = client.chat.completions.create(
                model="llama3.1-8b",
                messages=[
                    {"role": "system", "content": COMPRESS_SYSTEM},
                    {"role": "user", "content": f"Here are all the research findings to consolidate:\n\n{truncated}\n\nPlease clean up and consolidate these findings."},
                ],
                max_tokens=2000,
            )
            compressed = response.choices[0].message.content.strip()
            return {
                "compressed_research": compressed,
                "raw_notes": [raw_text[:2000]],  # Keep snippet of raw for supervisor
            }
        except Exception as e:
            if attempt < max_attempts - 1:
                char_limit = int(char_limit * 0.7)  # Reduce by 30% each retry
                continue
            # Final fallback
            return {
                "compressed_research": f"## Research Findings\n\nRaw findings (compression failed):\n\n{raw_text[:1500]}",
                "raw_notes": [raw_text[:1000]],
            }

    return {
        "compressed_research": "Error: Could not compress research after max retries.",
        "raw_notes": [],
    }
