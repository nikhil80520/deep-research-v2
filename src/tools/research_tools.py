from langchain_core.tools import tool
from src.graph.state import ConductResearch, ResearchComplete


@tool
def conduct_research(research_topic: str) -> str:
    """Delegate a research task to a sub-researcher agent.
    
    Args:
        research_topic: Detailed description of what to research.
    
    Returns:
        Confirmation string.
    """
    return f"Research task queued: {research_topic}"


@tool
def research_complete() -> str:
    """Signal that all research is complete and report can be written.
    
    Returns:
        Completion signal string.
    """
    return "Research marked as complete."
