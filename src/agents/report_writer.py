import os
from langchain_core.messages import AIMessage
from src.graph.state import AgentState


REPORT_SYSTEM = """You are a research report writer. Synthesize all research findings into a comprehensive, well-structured report.

Format requirements:
- Use # for title, ## for sections, ### for subsections
- Include inline citations as [1], [2], etc.
- End with a ### Sources section listing all URLs
- Be comprehensive — users expect deep research answers
- Write in the same language as the user's original query
- Do NOT refer to yourself as the writer"""


def _get_client():
    from cerebras.cloud.sdk import Cerebras
    return Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))


async def final_report_generation(state: AgentState, config=None) -> dict:
    """Synthesize all research findings into a final report with token limit retry."""
    notes = state.get("notes", [])
    research_brief = state.get("research_brief", "")
    findings = "\n\n---\n\n".join(notes) if notes else "No research findings available."

    client = _get_client()
    max_retries = 3
    char_limit = 8000  # chars of findings to include

    for attempt in range(max_retries):
        try:
            truncated_findings = findings[:char_limit]
            response = client.chat.completions.create(
                model="llama3.1-8b",
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM},
                    {"role": "user", "content": (
                        f"Research Brief:\n{research_brief}\n\n"
                        f"Research Findings:\n{truncated_findings}\n\n"
                        f"Write a comprehensive research report based on these findings."
                    )},
                ],
                max_tokens=3000,
            )
            report = response.choices[0].message.content.strip()

            return {
                "final_report": report,
                "messages": [AIMessage(content=report)],
            }

        except Exception as e:
            if attempt < max_retries - 1:
                char_limit = int(char_limit * 0.7)
                continue
            # Final fallback
            fallback = f"# Research Report\n\n{findings[:2000]}\n\n*Note: Full report generation failed. Raw findings above.*"
            return {
                "final_report": fallback,
                "messages": [AIMessage(content=fallback)],
            }

    return {
        "final_report": "Error: Report generation failed after maximum retries.",
        "messages": [AIMessage(content="Report generation failed.")],
    }
