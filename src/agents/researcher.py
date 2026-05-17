import json
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


def _parse_tool_calls(response_message) -> list[dict]:
    """Extract tool calls from a model response."""
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


async def researcher_node(state: ResearcherState, config=None) -> ResearcherOutputState:
    """Run the researcher ReAct loop and return compressed findings."""
    from src.config.configuration import Configuration

    configurable = Configuration.from_config(config)
    client = _get_client(configurable)
    topic = state.get("research_topic", "")
    max_calls = state.get("max_researcher_tool_calls", 10)

    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM},
        {"role": "user", "content": f"Research this topic thoroughly:\n\n{topic}"},
    ]

    search_count = 0
    max_searches = 5
    done = False

    for _ in range(max_calls):
        if done:
            break

        try:
            response = client.converse(
                modelId=configurable.llm_model,
                system=[{"text": RESEARCH_SYSTEM}],
                messages=_openai_messages_to_bedrock(messages),
                toolConfig={
                    "tools": _bedrock_tools([SEARCH_TOOL_DEF, THINK_TOOL_DEF, FINISH_TOOL_DEF]),
                    "toolChoice": {"auto": {}},
                },
                inferenceConfig={"maxTokens": 1000},
            )
        except Exception as e:
            print(f"  ⚠️  Researcher LLM call failed: {e}")
            break

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

        if tool_calls:
            tool_names = ", ".join(tc.get("name", "") for tc in tool_calls)
            print(f"🔧 Researcher tool calls: {tool_names}")

        # Add assistant message to history (with tool_calls if any)
        assistant_msg = {
            "role": "assistant",
            "content": msg_content or "",
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls → LLM is done
        if not tool_calls:
            break

        # Execute each tool call and append tool response
        for tc in tool_calls:
            result = ""

            if tc["name"] == "finish_research":
                result = "Research complete."
                done = True
            elif tc["name"] == "tavily_search":
                args = tc["args"]
                queries = args.get("queries", [])
                # Handle string input
                if isinstance(queries, str):
                    try:
                        queries = json.loads(queries.replace("'", '"'))
                    except Exception:
                        queries = [queries]
                # Flatten nested lists (LLM sometimes returns [['q1'], ['q2']])
                flat_queries = []
                for q in queries:
                    if isinstance(q, list):
                        flat_queries.extend([str(x) for x in q])
                    else:
                        flat_queries.append(str(q))
                args["queries"] = flat_queries
                if flat_queries:
                    print(f"🔍 Search queries: {', '.join(flat_queries)[:300]}")
                result = await tavily_search.ainvoke(args)
                print(f"✅ Search complete ({len(str(result))} chars)")
                # Print discovered URLs for the dashboard source cards
                import re as _re
                for _url_m in _re.finditer(r"URL:\s*(https?://\S+)", str(result)):
                    print(f"🌐 Source: {_url_m.group(1)}")
                search_count += 1
            elif tc["name"] == "think_tool":
                reflection = tc.get("args", {}).get("reflection", "")
                if reflection:
                    print(f"💭 Thinking: {reflection[:200]}")
                result = think_tool.invoke(tc["args"])
            else:
                result = f"Unknown tool: {tc['name']}"

            # Always append tool response for every tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)[:3000],
            })

        # Auto-stop after max searches
        if search_count >= max_searches:
            done = True

    # Convert raw messages to LangChain message types for compression
    langchain_messages = []
    for m in messages[1:]:  # Skip system message
        if m["role"] == "assistant":
            langchain_messages.append(AIMessage(content=m.get("content", "") or ""))
        elif m["role"] == "tool":
            langchain_messages.append(ToolMessage(
                content=m.get("content", ""),
                tool_call_id=m.get("tool_call_id", ""),
                name=m.get("name", "tool")
            ))
        else:
            langchain_messages.append(HumanMessage(content=m.get("content", "")))

    # Compress research findings (pass config through)
    compressed = await compress_research({
        "researcher_messages": langchain_messages
    }, config)

    return {
        "compressed_research": compressed.get("compressed_research", "No findings."),
        "raw_notes": [compressed.get("raw_notes", "")],
    }
