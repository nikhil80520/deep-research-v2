from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string

from src.graph.state import AgentState


FINAL_REPORT_PROMPT = """Based on research, create a comprehensive report:

<Research Brief>
{research_brief}
</Research Brief>

<Messages>
{messages}
</Messages>

<Findings>
{findings}
</Findings>

Today: {date}

Requirements:
1. Well-organized with ## headings
2. Include specific facts from research
3. Reference sources as [Title](URL)
4. Comprehensive - users expect detailed answers
5. Sources section at end

Citation Rules:
- Number sequentially: [1], [2], [3]...
- End with ### Sources list
"""


async def final_report_generation(state: AgentState, config=None) -> dict:
    """Synthesize all research findings into a final report using official prompt."""
    from src.config.configuration import Configuration
    from datetime import datetime

    configurable = Configuration.from_config(config)
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)

    model = configurable.get_model()

    # Token limit retry
    for attempt in range(3):
        try:
            prompt = FINAL_REPORT_PROMPT.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=datetime.now().strftime("%a %b %d, %Y")
            )

            response = await model.ainvoke([HumanMessage(content=prompt)])
            return {
                "final_report": str(response.content),
                "messages": [AIMessage(content=str(response.content))],
                "notes": [],
            }
        except Exception as e:
            if "token" in str(e).lower() or "context" in str(e).lower():
                # 10% truncate
                findings = findings[:int(len(findings) * 0.9)]
                continue
            # Non-token error — fallback
            fallback = f"# Research Report\n\n{findings[:2000]}\n\n*Note: Report generation failed: {e}*"
            return {
                "final_report": fallback,
                "messages": [AIMessage(content=fallback)],
            }

    return {
        "final_report": "Max retries exceeded",
        "messages": [AIMessage(content="Max retries exceeded")],
    }
