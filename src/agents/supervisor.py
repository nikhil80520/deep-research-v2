import json
import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from src.graph.state import SupervisorState
from src.tools.think import think_tool


SUPERVISOR_SYSTEM = """You are a research supervisor. Your job is to delegate research tasks to sub-researchers and then signal when research is complete.

Available tools:
1. conduct_research(research_topic) — delegate to a sub-researcher
2. think_tool(reflection) — plan strategy BEFORE delegating, assess gaps AFTER
3. research_complete() — call when all research is done

Rules:
- ALWAYS use think_tool BEFORE calling conduct_research to plan
- ALWAYS use think_tool AFTER results to assess gaps
- Bias towards single researcher unless clear parallelization opportunity
- Be explicit in research_topic — no acronyms, full context
- Call research_complete when satisfied
- Stop after {max_iterations} total tool calls"""

CONDUCT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "conduct_research",
        "description": "Delegate a research task to a sub-researcher agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": "Detailed description of what to research. No acronyms. Be explicit."
                }
            },
            "required": ["research_topic"]
        }
    }
}

THINK_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "think_tool",
        "description": "Reflect before/after delegating research. Call ALONE.",
        "parameters": {
            "type": "object",
            "properties": {
                "reflection": {"type": "string", "description": "Your strategic reflection."}
            },
            "required": ["reflection"]
        }
    }
}

COMPLETE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "research_complete",
        "description": "Call this when all research is complete.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
}


def _get_client(configurable):
    import boto3
    return boto3.client(
        "bedrock-runtime",
        region_name=configurable.aws_region,
        aws_access_key_id=configurable.aws_access_key_id or None,
        aws_secret_access_key=configurable.aws_secret_access_key or None,
        aws_session_token=configurable.aws_session_token or None,
    )


def _bedrock_tools(tool_defs: list[dict]) -> list[dict]:
    tools = []
    for tool_def in tool_defs:
        func = tool_def.get("function", {})
        tools.append({
            "toolSpec": {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "inputSchema": {"json": func.get("parameters", {})},
            }
        })
    return tools


def _openai_messages_to_bedrock(messages: list[dict]) -> list[dict]:
    bedrock_messages = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            content = []
            text = msg.get("content") or ""
            if text:
                content.append({"text": text})

            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_use_id = tc.get("id", "")
                try:
                    tool_input = json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments", {})
                except Exception:
                    tool_input = {}
                content.append({
                    "toolUse": {
                        "toolUseId": tool_use_id,
                        "name": func.get("name", ""),
                        "input": tool_input,
                    }
                })

            bedrock_messages.append({
                "role": "assistant",
                "content": content or [{"text": ""}],
            })
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_text = msg.get("content", "")
            bedrock_messages.append({
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": [{"text": str(tool_text)}],
                            "status": "success",
                        }
                    }
                ],
            })
        else:
            bedrock_messages.append({
                "role": "user",
                "content": [{"text": str(msg.get("content", ""))}],
            })
    return bedrock_messages


def _parse_tool_calls(msg) -> list[dict]:
    tool_calls = []
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except Exception:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    return tool_calls


async def supervisor_node(state: SupervisorState, config=None) -> Command[Literal["supervisor_tools", "__end__"]]:
    """Supervisor LLM decides what research to delegate."""
    from src.config.configuration import Configuration

    configurable = Configuration.from_config(config)
    client = _get_client(configurable)
    research_brief = state.get("research_brief", "")
    if isinstance(research_brief, dict):
        research_brief = json.dumps(research_brief)
    elif not isinstance(research_brief, str):
        research_brief = str(research_brief)
    max_iterations = 6

    supervisor_messages = state.get("supervisor_messages", [])
    iterations = state.get("research_iterations", 0)

    # Build message history for Bedrock
    messages = [
        {"role": "user", "content": research_brief},
    ]

    # Add conversation history
    for msg in supervisor_messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content) if isinstance(content, dict) else str(content)
            msg_copy = {k: (content if k == "content" else v) for k, v in msg.items()}
            messages.append(msg_copy)
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content or ""})
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": content})
        elif isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})

    try:
        response = client.converse(
            modelId=configurable.llm_model,
            system=[{"text": SUPERVISOR_SYSTEM.format(max_iterations=max_iterations)}],
            messages=_openai_messages_to_bedrock(messages),
            toolConfig={
                "tools": _bedrock_tools([CONDUCT_TOOL_DEF, THINK_TOOL_DEF, COMPLETE_TOOL_DEF]),
                "toolChoice": {"auto": {}},
            },
            inferenceConfig={"maxTokens": 1000},
        )
    except Exception as e:
        print(f"\n⚠️  Supervisor LLM call failed: {e}")
        return Command(goto="__end__", update={
            "research_brief": research_brief,
        })

    content = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = []
    tool_calls = []
    for item in content:
        if "text" in item:
            text_parts.append(item["text"])
        if "toolUse" in item:
            tool = item["toolUse"]
            tool_calls.append({
                "id": tool.get("toolUseId", ""),
                "name": tool.get("name", ""),
                "args": tool.get("input", {}) or {},
            })

    msg_content = "\n".join(t for t in text_parts if t).strip()

    # If no tool calls → end
    if not tool_calls:
        return Command(goto="__end__", update={
            "research_brief": research_brief,
        })

    # If research_complete called → end
    if any(tc["name"] == "research_complete" for tc in tool_calls):
        return Command(goto="__end__", update={
            "research_brief": research_brief,
        })

    new_msg = {"role": "assistant", "content": msg_content or "", "tool_calls": [
        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
        for tc in tool_calls
    ]}

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [new_msg],
            "research_iterations": iterations + 1,
        }
    )


async def supervisor_tools_node(state: SupervisorState, config=None) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor tool calls — run researchers in parallel."""
    from src.agents.researcher import researcher_node

    supervisor_messages = state.get("supervisor_messages", [])
    iterations = state.get("research_iterations", 0)
    max_iterations = 6

    # Get last assistant message
    last_msg = None
    for msg in reversed(supervisor_messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_msg = msg
            break

    if not last_msg:
        return Command(goto="__end__", update={
            "research_brief": state.get("research_brief", ""),
        })

    tool_calls = last_msg.get("tool_calls") or []

    # Check exit conditions
    no_tool_calls = not tool_calls
    over_limit = iterations > max_iterations
    research_done = any(tc["function"]["name"] == "research_complete" for tc in tool_calls)

    if no_tool_calls or over_limit or research_done:
        if over_limit:
            print(f"⏹️  Supervisor hit iteration limit ({max_iterations}), finishing research.")
        return Command(goto="__end__", update={
            "research_brief": state.get("research_brief", ""),
        })

    tool_messages = []
    all_new_notes = []
    results = []

    # Separate think_tool calls from conduct_research calls
    think_calls = [tc for tc in tool_calls if tc["function"]["name"] == "think_tool"]
    conduct_calls = [tc for tc in tool_calls if tc["function"]["name"] == "conduct_research"]

    # Handle think_tool
    for tc in think_calls:
        try:
            args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
        except Exception:
            args = {}
        reflection = args.get("reflection", "")
        print(f"💭 Thinking: {reflection[:100]}...")
        result = think_tool.invoke(args)
        tool_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})

    # Handle conduct_research — run in parallel
    if conduct_calls:
        max_concurrent = 5
        allowed = conduct_calls[:max_concurrent]

        async def run_researcher(tc):
            try:
                args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
            except Exception:
                args = {}
            topic = args.get("research_topic", "")
            print(f"🔍 Researcher starting: {topic[:80]}...")
            result = await researcher_node({
                "research_topic": topic,
                "researcher_messages": [],
                "tool_call_iterations": 0,
                "compressed_research": "",
                "raw_notes": [],
                "max_researcher_tool_calls": 10,
            })
            return tc["id"], result

        results = await asyncio.gather(*[run_researcher(tc) for tc in allowed])

        for tool_call_id, result in results:
            compressed = result.get("compressed_research", "No findings.")
            raw = result.get("raw_notes", [])
            print(f"✅ Researcher done: {len(compressed)} chars")
            all_new_notes.extend([compressed])
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": compressed[:2000],
            })

    # Collect raw_notes from all researchers
    all_raw_notes = [r for tc_id, res in results for r in res.get("raw_notes", [])]

    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": tool_messages,
            "notes": all_new_notes,
            "raw_notes": all_raw_notes,
            "research_iterations": iterations + 1,
        }
    )


