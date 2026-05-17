import streamlit as st
import asyncio
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research, get_history
from src.ui.trace_events import TraceEvent, parse_trace_line, determine_pipeline_stage
from src.ui.dashboard_renderer import (
    render_command_center,
    render_activity_feed,
    render_source_cards,
    render_knowledge_sidebar,
    render_final_report,
    render_methodology_accordion,
    inject_auto_scroll,
)
init_db()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "last_trace" not in st.session_state:
    st.session_state.last_trace = ""
if "trace_events" not in st.session_state:
    st.session_state.trace_events = []
if "is_research_complete" not in st.session_state:
    st.session_state.is_research_complete = False
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_brief" not in st.session_state:
    st.session_state.last_brief = ""
if "last_research_id" not in st.session_state:
    st.session_state.last_research_id = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px;">
            ⚙️ CONTROL PANEL
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_id = st.text_input("User ID", value="default_user", label_visibility="collapsed",
                            placeholder="Enter User ID...")

    col_a, col_b = st.columns(2)
    with col_a:
        load_btn = st.button("📚 History", use_container_width=True)
    with col_b:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.awaiting_clarification = False
            st.session_state.current_query = ""
            st.session_state.trace_events = []
            st.session_state.is_research_complete = False
            st.session_state.last_report = None
            st.session_state.last_brief = ""
            st.session_state.last_research_id = None
            st.session_state.last_trace = ""
            st.rerun()

    if load_btn:
        history = get_history(user_id, limit=10)
        if history:
            for h in history:
                with st.expander(f"#{h['id']} — {h['query'][:50]}..."):
                    st.caption(h["created_at"])
                    st.write(h["final_report"])
        else:
            st.info("No history found.")

    st.markdown("<hr style='border-color: rgba(99,102,241,0.1); margin: 16px 0;'>",
                unsafe_allow_html=True)

    # Knowledge Graph sidebar (rendered from events)
    render_knowledge_sidebar(
        st.session_state.trace_events,
        st.session_state.last_trace,
    )

# ── Command Center Header ─────────────────────────────────────────────────────
render_command_center(
    events=st.session_state.trace_events,
    is_running=False,
    is_complete=st.session_state.is_research_complete,
)

# ── Query Input ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    if st.session_state.awaiting_clarification:
        query = st.text_area(
            "Your clarification",
            placeholder="Provide more details...",
            height=80,
            key="clarification_input",
            label_visibility="collapsed",
        )
    else:
        query = st.text_area(
            "Research Query",
            placeholder="What are the latest advances in quantum computing in 2025?",
            height=80,
            key="query_input",
            label_visibility="collapsed",
        )

with col2:
    st.write("")
    if st.session_state.awaiting_clarification:
        run_btn = st.button("💬 Submit", type="primary", use_container_width=True)
    else:
        run_btn = st.button("🚀 Research", type="primary", use_container_width=True)

# ── Conversation History ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# ── Show Previous Complete Results ─────────────────────────────────────────────
if st.session_state.is_research_complete and st.session_state.last_report:
    # Show methodology accordion (collapsed trace)
    render_methodology_accordion(
        st.session_state.trace_events,
        st.session_state.last_trace,
    )

    # Show final report
    render_final_report(
        st.session_state.last_report,
        st.session_state.last_brief,
        st.session_state.last_research_id,
    )

    st.download_button(
        label="⬇️ Download Report",
        data=st.session_state.last_report,
        file_name=f"research_{st.session_state.last_research_id or 'report'}.md",
        mime="text/markdown",
    )

# ── Active Intelligence Feed (shown when no complete results, or during run) ──
elif st.session_state.trace_events and not st.session_state.is_research_complete:
    render_activity_feed(st.session_state.trace_events, show_shimmer=False)
    render_source_cards(st.session_state.last_trace)

elif not st.session_state.is_research_complete:
    # Empty state
    render_activity_feed([], show_shimmer=False)

# ── Trace Containers for live updates ──────────────────────────────────────────
trace_container = st.empty()
feed_container = st.empty()

if run_btn and query.strip():
    # Reset previous results
    st.session_state.is_research_complete = False
    st.session_state.last_report = None
    st.session_state.last_brief = ""
    st.session_state.last_research_id = None
    st.session_state.trace_events = []
    st.session_state.last_trace = ""

    with st.spinner(""):
        status = st.empty()

        async def run_research():
            class _DashboardTraceWriter(io.StringIO):
                """Intercepts print() output, parses into TraceEvents,
                and renders the live dashboard feed."""
                def __init__(self, feed_placeholder, trace_placeholder):
                    super().__init__()
                    self._feed_placeholder = feed_placeholder
                    self._trace_placeholder = trace_placeholder
                    self._events = []
                    self._line_buffer = ""

                def write(self, s):
                    written = super().write(s)
                    # Buffer lines (print may call write multiple times per line)
                    self._line_buffer += s
                    while "\n" in self._line_buffer:
                        line, self._line_buffer = self._line_buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            event = parse_trace_line(line)
                            if event:
                                self._events.append(event)
                                st.session_state.trace_events = list(self._events)
                                self._render_feed()
                    return written

                def _render_feed(self):
                    """Re-render the live activity feed."""
                    with self._feed_placeholder.container():
                        # Re-render command center in running state
                        render_command_center(
                            events=self._events,
                            is_running=True,
                            is_complete=False,
                        )
                        render_activity_feed(self._events, show_shimmer=True)

                        # Auto-scroll
                        inject_auto_scroll()

                def get_events(self):
                    return list(self._events)

            log_buffer = _DashboardTraceWriter(feed_container, trace_container)

            if st.session_state.awaiting_clarification:
                status.info("🧠 Processing clarification...")
                st.session_state.messages.append(HumanMessage(content=query))
            else:
                status.info("🎯 Analyzing intent...")
                user_content = query
                st.session_state.current_query = query
                st.session_state.messages = [HumanMessage(content=user_content)]

            status.info("📋 Writing research brief...")
            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                result = await deep_research_graph.ainvoke({
                    "messages": st.session_state.messages,
                    "user_id": user_id,
                    "supervisor_messages": [],
                    "raw_notes": [],
                    "notes": [],
                    "final_report": "",
                    "research_brief": "",
                })
            return result, log_buffer.getvalue(), log_buffer.get_events()

        result, trace, events = asyncio.run(run_research())
        st.session_state.last_trace = trace.strip()
        st.session_state.trace_events = events

        # Clear status
        status.empty()

        # Check if we got a clarifying question
        last_message = result.get("messages", [])[-1] if result.get("messages") else None
        if last_message and hasattr(last_message, 'content') and last_message.content:
            if not result.get("final_report") and not result.get("research_brief"):
                # This is a clarification question
                st.session_state.messages.append(last_message)
                st.session_state.awaiting_clarification = True
                st.rerun()
            else:
                # Research completed
                st.session_state.awaiting_clarification = False

                final_report = result.get("final_report", "No report generated.")
                brief = result.get("research_brief", "")

                # Save
                rid = save_research(user_id, st.session_state.current_query or query, {
                    "research_brief": brief,
                    "final_report": final_report,
                    "notes": result.get("notes", []),
                })

                # Store in session for persistent display
                st.session_state.is_research_complete = True
                st.session_state.last_report = final_report
                st.session_state.last_brief = brief
                st.session_state.last_research_id = rid
                st.session_state.messages = []
                st.session_state.current_query = ""

                # Rerun to show final state
                st.rerun()

elif run_btn:
    st.warning("Please enter a research query.")
