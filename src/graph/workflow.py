from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState, AgentInputState, SupervisorState
from src.agents.clarifier import clarify_with_user
from src.agents.brief_writer import write_research_brief
from src.agents.supervisor import supervisor_node, supervisor_tools_node
from src.agents.report_writer import final_report_generation


def build_supervisor_subgraph():
    """Build the supervisor subgraph (supervisor ↔ supervisor_tools loop)."""
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("supervisor_tools", supervisor_tools_node)
    builder.add_edge(START, "supervisor")
    # Routing is handled via Command.goto in supervisor_node and supervisor_tools_node
    return builder.compile()


def build_main_graph():
    """Build the complete deep research agent graph."""
    supervisor_subgraph = build_supervisor_subgraph()

    builder = StateGraph(AgentState, input=AgentInputState)

    # Nodes
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)
    builder.add_node("research_supervisor", supervisor_subgraph)
    builder.add_node("final_report_generation", final_report_generation)

    # Edges
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("research_supervisor", "final_report_generation")
    builder.add_edge("final_report_generation", END)

    return builder.compile()


# Compiled graph — import this in API and run.py
deep_research_graph = build_main_graph()
