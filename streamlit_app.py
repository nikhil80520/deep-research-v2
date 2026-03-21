import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research, get_history
from src.memory.mem0_client import init_memory, add_memory, search_memory

init_db()
init_memory()

st.set_page_config(page_title="Deep Research Agent", page_icon="🔬", layout="wide")

st.title("🔬 Deep Research Agent v2")
st.caption("Supervisor → Parallel Researchers → Comprehensive Report")

# Initialize session state for multi-turn conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Sidebar — history
with st.sidebar:
    st.header("📚 Research History")
    user_id = st.text_input("User ID", value="default_user")
    if st.button("Load History"):
        history = get_history(user_id, limit=10)
        if history:
            for h in history:
                with st.expander(f"#{h['id']} — {h['query'][:50]}..."):
                    st.caption(h["created_at"])
                    st.write(h["final_report"])
        else:
            st.info("No history found.")

    # Reset conversation button
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.awaiting_clarification = False
        st.session_state.current_query = ""
        st.rerun()

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.awaiting_clarification:
        query = st.text_area(
            "Your clarification",
            placeholder="Provide more details...",
            height=100,
            key="clarification_input"
        )
    else:
        query = st.text_area(
            "Research Query",
            placeholder="What are the latest advances in quantum computing in 2025?",
            height=100,
            key="query_input"
        )

with col2:
    st.write("")
    st.write("")
    if st.session_state.awaiting_clarification:
        run_btn = st.button("💬 Submit Clarification", type="primary", use_container_width=True)
    else:
        run_btn = st.button("🚀 Run Deep Research", type="primary", use_container_width=True)

# Display conversation history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

if run_btn and query.strip():
    with st.spinner("Processing..."):
        status = st.empty()

        async def run_research():
            if st.session_state.awaiting_clarification:
                # Continue with clarification
                status.info("🧠 Processing clarification...")
                st.session_state.messages.append(HumanMessage(content=query))
            else:
                # New query
                status.info("🧠 Clarifying query...")
                memory_ctx = search_memory(user_id, query)

                user_content = query
                if memory_ctx:
                    user_content = f"{memory_ctx}\n\n---\n\nCurrent query: {query}"
                    status.info("💾 Found relevant past research — using as context...")

                st.session_state.current_query = query
                st.session_state.messages = [HumanMessage(content=user_content)]

            status.info("📋 Writing research brief...")
            result = await deep_research_graph.ainvoke({
                "messages": st.session_state.messages,
                "user_id": user_id,
                "supervisor_messages": [],
                "raw_notes": [],
                "notes": [],
                "final_report": "",
                "research_brief": "",
            })
            return result

        result = asyncio.run(run_research())

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
                add_memory(user_id, st.session_state.current_query or query, final_report[:300])

                # Show results
                st.success(f"✅ Research complete! (ID: #{rid})")

                if brief:
                    with st.expander("📋 Research Brief"):
                        st.write(brief)

                st.markdown("---")
                st.subheader("📄 Final Report")
                st.markdown(final_report)

                st.download_button(
                    label="⬇️ Download Report",
                    data=final_report,
                    file_name=f"research_{rid}.md",
                    mime="text/markdown",
                )

                # Clear session state for next query
                st.session_state.messages = []
                st.session_state.current_query = ""

elif run_btn:
    st.warning("Please enter a research query.")
