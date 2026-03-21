import os
import json
import re
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.graph.state import ResearcherState, ResearcherOutputState
from src.tools.search import tavily_search
from src.tools.think import think_tool
from src.agents.compressor import compress_research


RESEARCH_SYSTEM = """You are a focused research assistant. Research the given topic thoroughly.

Available tools:
1. tavily_search — search the web (provide 1-3 specific queries)
2. think_tool — reflect on findings after each search (NEVER in parallel with search)
3. finish_research — call when you have enough information

Research process:
1. Start with broad queries
2. After each search: use think_tool to assess findings
3. Do targeted follow-up searches for gaps
4. Stop when you have 3+ good sources OR after 5 searches
5. Call finish_research when done

IMPORTANT: think_tool must be called ALONE, never with other tools."""

FINISH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "finish_research",
        "description": "Call this when you have gathered enough information and are ready to finalize.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
}

SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries (1-3)"
                }
            },
            "required": ["queries"]
        }
    }
}

THINK_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "think_tool",
        "description": "Reflect on research progress. Call ALONE, not with other tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "reflection": {
                    "type": "string",
                    "description": "Your reflection on what you found and what to do next."
                }
            },
            "required": ["reflection"]
        }
    }
}


def _get_client():
    from cerebras.cloud.sdk import Cerebras
    return Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))


def _parse_tool_calls(response_message) -> list[dict]:
    """Extract tool calls from Cerebras response."""
    tool_calls = []
    if hasattr(response_message, "tool_calls") and response_message.tool_calls:
        for tc in response_message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except Exception:
                args = {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
            })
    return tool_calls


async def researcher_node(state: ResearcherState) -> ResearcherOutputState:
    """Run the researcher ReAct loop and return compressed findings."""
    client = _get_client()
    topic = state.get("research_topic", "")
    max_calls = state.get("max_researcher_tool_calls", 10)

    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM},
        {"role": "user", "content": f"Research this topic thoroughly:\n\n{topic}"},
    ]

    all_researcher_messages = list(state.get("researcher_messages", []))
    tool_calls_made = state.get("tool_call_iterations", 0)

    for _ in range(max_calls):
        try:
            response = client.chat.completions.create(
                model="llama3.1-8b",
                messages=messages,
                tools=[SEARCH_TOOL_DEF, THINK_TOOL_DEF, FINISH_TOOL_DEF],
                tool_choice="auto",
                max_tokens=1000,
            )
        except Exception as e:
            break

        msg = response.choices[0].message
        tool_calls = _parse_tool_calls(msg)

        # Add assistant message to history
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                for tc in tool_calls
            ] if tool_calls else None
        })

        if not tool_calls:
            break

        # Check for finish
        if any(tc["name"] == "finish_research" for tc in tool_calls):
            messages.append({"role": "tool", "tool_call_id": tool_calls[-1]["id"], "content": "Research complete."})
            break

        # Execute tools
        for tc in tool_calls:
            tool_call_made = True
            result = ""
            if tc["name"] == "tavily_search":
                result = await tavily_search.ainvoke(tc["args"])
            elif tc["name"] == "think_tool":
                result = think_tool.invoke(tc["args"])
            else:
                result = f"Unknown tool: {tc['name']}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)[:3000],
                "name": tc["name"],
            })

        tool_calls_made += 1

    # Store messages for compression
    state_with_messages = {**state, "researcher_messages": [
        AIMessage(content=m.get("content", "") or "") if m["role"] == "assistant"
        else ToolMessage(content=m.get("content", ""), tool_call_id=m.get("tool_call_id", ""), name=m.get("name", "tool"))
        if m["role"] == "tool"
        else HumanMessage(content=m.get("content", ""))
        for m in messages[1:]  # Skip system message
    ]}

    # Compress and return
    result = await compress_research(state_with_messages)
    return result
