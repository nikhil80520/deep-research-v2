from typing import Literal
from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.types import Command
from pydantic import BaseModel, Field


TRANSFORM_PROMPT = """You will be given messages from the user. 
Translate these into a detailed research brief.

<Messages>
{messages}
</Messages>

Today's date is {date}.

Guidelines:
1. Maximize specificity - include all user preferences
2. Fill unstated dimensions as open-ended
3. Use first person perspective
4. Specify preferred sources if relevant
"""


class ResearchBrief(BaseModel):
    research_brief: str = Field(
        description="Detailed research brief to guide the research."
    )


async def write_research_brief(state, config=None) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief using official prompt."""
    from src.config.configuration import Configuration
    from datetime import datetime

    configurable = Configuration.from_config(config)
    messages = state.get("messages", [])

    model = configurable.get_model(structured_output=ResearchBrief)

    prompt = TRANSFORM_PROMPT.format(
        messages=get_buffer_string(messages),
        date=datetime.now().strftime("%a %b %d, %Y")
    )

    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        brief = response.research_brief
    except Exception as e:
        print(f"⚠️  Brief generation failed: {e}")
        # Fallback to raw messages
        brief = get_buffer_string(messages) if messages else "No messages."

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
