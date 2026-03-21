from langchain_core.tools import tool


@tool(description="Strategic reflection tool. Use after every search to plan next steps.")
def think_tool(reflection: str) -> str:
    """Record a strategic reflection during research.

    Use after each search to assess:
    - What did I find?
    - What is still missing?
    - Should I search more or stop?

    NEVER call in parallel with search tools — always separate.

    Args:
        reflection: Your detailed reflection on progress and next steps.

    Returns:
        Confirmation string.
    """
    return f"Reflection recorded: {reflection}"
