import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph.workflow import deep_research_graph
from src.memory.database import init_db, save_research, get_history
from src.memory.mem0_client import init_memory, add_memory, search_memory

init_db()
init_memory()

st.set_page_config(page_title="Deep Research Agent", page_icon="🔬", layout="wide")

st.title("🔬 Deep Research Agent v2")
st.caption("Supervisor → Parallel Researchers → Comprehensive Report")

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

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area(
        "Research Query",
        placeholder="What are the latest advances in quantum computing in 2025?",
        height=100,
    )

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Deep Research", type="primary", use_container_width=True)

if run_btn and query.strip():
    with st.spinner("Initializing research pipeline..."):
        status = st.empty()
        report_area = st.empty()

        async def run_research():
            status.info("🧠 Clarifying query...")
            memory_ctx = search_memory(user_id, query)

            user_content = query
            if memory_ctx:
                user_content = f"{memory_ctx}\n\n---\n\nCurrent query: {query}"
                status.info("💾 Found relevant past research — using as context...")

            status.info("📋 Writing research brief...")
            result = await deep_research_graph.ainvoke({
                "messages": [HumanMessage(content=user_content)],
                "user_id": user_id,
                "supervisor_messages": [],
                "raw_notes": [],
                "notes": [],
                "final_report": "",
                "research_brief": "",
            })
            return result

        result = asyncio.run(run_research())

    final_report = result.get("final_report", "No report generated.")
    brief = result.get("research_brief", "")

    # Save
    rid = save_research(user_id, query, {
        "research_brief": brief,
        "final_report": final_report,
        "notes": result.get("notes", []),
    })
    add_memory(user_id, query, final_report[:300])

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

elif run_btn:
    st.warning("Please enter a research query.")
