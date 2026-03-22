from langchain_core.messages import HumanMessage, SystemMessage, filter_messages


COMPRESS_SYSTEM = """You are compressing research findings. 
Preserve ALL relevant information verbatim, just remove duplicates.
Include inline citations and a Sources section.

**List of Queries and Tool Calls Made**
**Fully Comprehensive Findings**  
**List of All Relevant Sources**

Today: {date}
"""

COMPRESS_HUMAN = """Clean up these research findings.
DO NOT summarize. Return raw info in cleaner format. Preserve everything."""


async def compress_research(state, config=None):
    """Compress researcher messages into clean findings with verbatim preservation."""
    from src.config.configuration import Configuration
    from datetime import datetime

    configurable = Configuration.from_config(config)
    researcher_messages = list(state.get("researcher_messages", []))

    # Append compression instruction
    researcher_messages = researcher_messages + [
        HumanMessage(content=COMPRESS_HUMAN)
    ]

    model = configurable.get_model()

    # Retry with truncation
    for attempt in range(3):
        try:
            system = COMPRESS_SYSTEM.format(
                date=datetime.now().strftime("%a %b %d, %Y")
            )
            response = await model.ainvoke(
                [SystemMessage(content=system)] + researcher_messages
            )

            # Extract raw notes from tool and AI messages
            raw = "\n".join([
                str(m.content)
                for m in filter_messages(
                    researcher_messages,
                    include_types=["tool", "ai"]
                )
            ])

            return {
                "compressed_research": str(response.content),
                "raw_notes": raw
            }
        except Exception as e:
            if attempt < 2:
                # Truncate oldest messages
                researcher_messages = researcher_messages[2:]
                continue
            return {
                "compressed_research": "Compression failed",
                "raw_notes": ""
            }
