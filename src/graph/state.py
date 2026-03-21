import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.graph.reducers import override_reducer


# ── Pydantic Structured Outputs ───────────────────────────────────────────────

class ConductResearch(BaseModel):
    """Supervisor calls this to delegate a research task to a sub-researcher."""
    research_topic: str = Field(
        description="The specific topic to research. Be explicit and detailed — no acronyms."
    )

class ResearchComplete(BaseModel):
    """Supervisor calls this when satisfied with all research findings."""
    pass

class ClarifyWithUser(BaseModel):
    """Result of the clarification analysis."""
    need_clarification: bool = Field(description="Whether to ask the user a question.")
    question: str = Field(description="The clarifying question to ask (empty if not needed).")
    verification: str = Field(description="Acknowledgement message if proceeding (empty if asking).")

class ResearchQuestion(BaseModel):
    """Structured research brief derived from user messages."""
    research_brief: str = Field(description="Detailed research brief guiding all downstream research.")


# ── Graph States ───────────────────────────────────────────────────────────────

class AgentInputState(MessagesState):
    """Entry state — only user messages."""
    pass

class AgentState(MessagesState):
    """Top-level agent state."""
    supervisor_messages: Annotated[list, override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add]
    final_report: str
    user_id: str

class SupervisorState(TypedDict):
    """State for the supervisor subgraph."""
    supervisor_messages: Annotated[list, override_reducer]
    research_brief: str
    notes: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], operator.add]
    research_iterations: int

class ResearcherState(TypedDict):
    """State for individual researcher subgraphs."""
    researcher_messages: Annotated[list, operator.add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]

class ResearcherOutputState(TypedDict):
    """Output from a researcher subgraph back to supervisor."""
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]
