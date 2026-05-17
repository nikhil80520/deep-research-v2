"""
Dashboard Renderer — uses st.html() to bypass Streamlit's markdown parser
which was stripping/escaping nested <div> tags even with unsafe_allow_html.
"""

import html as html_lib
from typing import Optional

import streamlit as st

from src.ui.trace_events import TraceEvent, determine_pipeline_stage, extract_urls_from_text

_ICONS = {
    "thinking": "🧠", "search_query": "🔍", "search_complete": "📥",
    "researcher_start": "🚀", "researcher_done": "✅", "tool_calls": "🔧",
    "source_url": "🌐",
    "supervisor_error": "⚠️", "researcher_error": "⚠️", "iteration_limit": "⏹️",
    "brief_writing": "📋", "clarifying": "🎯", "brief_error": "⚠️", "info": "ℹ️",
}

_CARD_CLASSES = {
    "thinking": "thinking", "search_query": "search", "search_complete": "search",
    "researcher_start": "researcher", "researcher_done": "researcher",
    "tool_calls": "tool", "source_url": "search",
    "supervisor_error": "error", "researcher_error": "error",
    "iteration_limit": "info", "brief_writing": "info", "clarifying": "info",
    "brief_error": "error", "info": "info",
}

def _esc(text: str) -> str:
    return html_lib.escape(str(text))


# ── Inline CSS (embedded in each st.html call so styles work) ──────────────────

def _get_inline_css() -> str:
    """Return CSS to embed inside st.html blocks so they are self-contained."""
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:transparent;color:#f1f5f9}
.command-center{background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(30,27,75,0.7));border:1px solid rgba(99,102,241,0.15);border-radius:16px;padding:24px 32px;margin-bottom:24px;position:relative;overflow:hidden}
.command-center::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#6366f1,transparent);opacity:0.6}
.command-center-title{display:flex;align-items:center;gap:16px;margin-bottom:20px}
.command-center-title h1{font-size:1.6rem;font-weight:700;background:linear-gradient(135deg,#f1f5f9,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;line-height:1.2}
.subtitle{font-size:0.8rem;color:#64748b;font-weight:400;letter-spacing:0.5px}
.pulse-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:100px;font-size:0.78rem;font-weight:600;letter-spacing:0.3px;text-transform:uppercase}
.pulse-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.pulse-badge.idle{background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.25)}
.pulse-badge.idle .pulse-dot{background:#10b981}
.pulse-badge.researching{background:rgba(99,102,241,0.12);color:#818cf8;border:1px solid rgba(99,102,241,0.3)}
.pulse-badge.researching .pulse-dot{background:#6366f1;animation:pulse-anim 1.5s ease-in-out infinite}
.pulse-badge.synthesizing{background:rgba(245,158,11,0.12);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
.pulse-badge.synthesizing .pulse-dot{background:#f59e0b;animation:pulse-anim 1.2s ease-in-out infinite}
.pulse-badge.complete{background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.25)}
.pulse-badge.complete .pulse-dot{background:#10b981}
@keyframes pulse-anim{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.6;transform:scale(1.6);box-shadow:0 0 8px 2px currentColor}}
.pipeline-breadcrumb{display:flex;align-items:center;gap:0;flex-wrap:wrap}
.pipeline-step{display:flex;align-items:center;gap:8px;padding:8px 6px;font-size:0.75rem;font-weight:500;color:#64748b}
.step-dot{width:10px;height:10px;border-radius:50%;border:2px solid #64748b;background:transparent;flex-shrink:0}
.pipeline-step.active{color:#818cf8}
.pipeline-step.active .step-dot{border-color:#6366f1;background:#6366f1;box-shadow:0 0 10px rgba(99,102,241,0.5);animation:pulse-anim 2s ease-in-out infinite}
.pipeline-step.completed{color:#34d399}
.pipeline-step.completed .step-dot{border-color:#10b981;background:#10b981;box-shadow:0 0 8px rgba(16,185,129,0.3)}
.pipeline-arrow{color:#64748b;font-size:0.7rem;margin:0 2px;opacity:0.5}
.pipeline-arrow.active{color:#818cf8;opacity:1}
.action-card{background:rgba(15,23,42,0.75);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:16px 20px;margin-bottom:10px;position:relative;overflow:hidden}
.action-card::before{content:'';position:absolute;top:0;left:0;bottom:0;width:3px;border-radius:3px 0 0 3px}
.action-card.thinking::before{background:linear-gradient(180deg,#a855f7,#c084fc)}
.action-card.search::before{background:linear-gradient(180deg,#6366f1,#818cf8)}
.action-card.researcher::before{background:linear-gradient(180deg,#10b981,#34d399)}
.action-card.error::before{background:linear-gradient(180deg,#f43f5e,#fb7185)}
.action-card.info::before{background:linear-gradient(180deg,#64748b,#94a3b8)}
.action-card.tool::before{background:linear-gradient(180deg,#f59e0b,#fbbf24)}
.card-header{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.card-icon{font-size:1.1rem;width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.03)}
.card-title{font-size:0.85rem;font-weight:600;color:#f1f5f9;flex:1}
.card-timestamp{font-size:0.68rem;color:#64748b;font-family:'JetBrains Mono',monospace}
.card-detail{font-size:0.8rem;color:#94a3b8;line-height:1.5;padding-left:38px}
.card-detail code{font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(99,102,241,0.1);padding:2px 6px;border-radius:4px;color:#818cf8}
.query-tag{display:inline-block;padding:4px 10px;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:6px;font-size:0.73rem;color:#818cf8;font-family:'JetBrains Mono',monospace;margin:3px 2px 3px 38px}
.data-badge{display:inline-flex;align-items:center;padding:2px 8px;background:rgba(16,185,129,0.12);border-radius:100px;font-size:0.65rem;font-weight:600;color:#34d399;font-family:'JetBrains Mono',monospace;border:1px solid rgba(16,185,129,0.2)}
.source-chip{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.15);border-radius:8px;margin:4px 4px 4px 0}
.source-chip img{width:16px;height:16px;border-radius:3px}
.source-chip .url-breadcrumb{font-size:0.72rem;color:#818cf8;font-family:'JetBrains Mono',monospace}
.source-chip a{color:#818cf8;text-decoration:none}
.source-chip a:hover{text-decoration:underline}
.feed-title{display:flex;align-items:center;gap:10px;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid rgba(99,102,241,0.15)}
.feed-title h3{font-size:0.95rem;font-weight:700;color:#f1f5f9;margin:0}
.count-badge{background:rgba(99,102,241,0.12);color:#818cf8;font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:100px;font-family:'JetBrains Mono',monospace}
.sidebar-section{background:rgba(15,23,42,0.75);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:16px;margin-bottom:12px}
.sidebar-section-title{font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(99,102,241,0.06)}
.stat-row:last-child{border-bottom:none}
.stat-label{font-size:0.75rem;color:#94a3b8}
.stat-value{font-size:0.85rem;font-weight:700;color:#f1f5f9;font-family:'JetBrains Mono',monospace}
.stat-value.accent-blue{color:#818cf8}
.stat-value.accent-emerald{color:#34d399}
.stat-value.accent-amber{color:#fbbf24}
.query-item{display:flex;align-items:flex-start;gap:8px;padding:8px 0;border-bottom:1px solid rgba(99,102,241,0.06)}
.query-item:last-child{border-bottom:none}
.query-status{flex-shrink:0;font-size:0.8rem;margin-top:1px}
.query-text{font-size:0.73rem;color:#94a3b8;line-height:1.4}
.shimmer-card{background:linear-gradient(90deg,rgba(15,23,42,0.75) 25%,rgba(99,102,241,0.08) 50%,rgba(15,23,42,0.75) 75%);background-size:200% 100%;animation:shimmer 1.8s ease-in-out infinite;border-radius:12px;padding:16px 20px;margin-bottom:10px;border:1px solid rgba(99,102,241,0.15)}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.shimmer-line{height:12px;background:rgba(99,102,241,0.06);border-radius:6px;margin-bottom:8px}
.shimmer-line.short{width:40%}
.shimmer-line.medium{width:65%}
.shimmer-line.long{width:85%}
.success-banner{background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(16,185,129,0.05));border:1px solid rgba(16,185,129,0.25);border-radius:12px;padding:16px 24px;display:flex;align-items:center;gap:12px;margin-bottom:16px}
.success-banner .icon{font-size:1.3rem}
.success-banner .text{font-size:0.9rem;font-weight:600;color:#34d399}
.success-banner .subtext{font-size:0.75rem;color:#64748b;margin-top:2px}
.final-report-container{background:linear-gradient(135deg,rgba(15,23,42,0.6),rgba(30,27,75,0.3));border:1px solid rgba(99,102,241,0.15);border-radius:16px;padding:32px 40px;margin-top:16px;position:relative;overflow:hidden}
.final-report-container::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#10b981,#6366f1,#a855f7)}
.empty-state{text-align:center;padding:30px;background:rgba(15,23,42,0.75);border:1px solid rgba(99,102,241,0.15);border-radius:12px}
.empty-state .icon{font-size:1.5rem;margin-bottom:8px}
.empty-state .msg{color:#64748b;font-size:0.85rem}
</style>"""


# ── Command Center ─────────────────────────────────────────────────────────────

def render_command_center(events, is_running=False, is_complete=False):
    if is_complete:
        badge_class, badge_text = "complete", "Complete"
    elif is_running:
        stage = determine_pipeline_stage(events)
        badge_class = "synthesizing" if stage >= 2 else "researching"
        badge_text = "Synthesizing..." if stage >= 2 else "Researching..."
    else:
        badge_class, badge_text = "idle", "Ready"

    pipeline_stage = determine_pipeline_stage(events) if events else -1
    if is_complete:
        pipeline_stage = 3

    stages = ["Intent Analysis", "Parallel Extraction", "Synthesis", "Final Report"]
    steps = ""
    for i, name in enumerate(stages):
        if i > 0:
            ac = "active" if i <= pipeline_stage else ""
            steps += f'<span class="pipeline-arrow {ac}">→</span>'
        if is_complete or i < pipeline_stage:
            sc = "completed"
        elif i == pipeline_stage:
            sc = "active"
        else:
            sc = ""
        steps += f'<div class="pipeline-step {sc}"><span class="step-dot"></span><span>{name}</span></div>'

    st.html(
        _get_inline_css()
        + f'<div class="command-center">'
        + f'<div class="command-center-title">'
        + f'<div><h1>🔬 Deep Research Agent</h1><div class="subtitle">Supervisor → Parallel Researchers → Comprehensive Report</div></div>'
        + f'<div style="margin-left:auto;"><span class="pulse-badge {badge_class}"><span class="pulse-dot"></span>{badge_text}</span></div>'
        + f'</div>'
        + f'<div class="pipeline-breadcrumb">{steps}</div>'
        + f'</div>'
    )


# ── Action Cards ───────────────────────────────────────────────────────────────

def _render_action_card(event):
    icon = _ICONS.get(event.event_type, "ℹ️")
    cc = _CARD_CLASSES.get(event.event_type, "info")
    ts = event.timestamp.strftime("%H:%M:%S")
    title = _esc(event.title)
    detail = _esc(event.detail)
    extra = ""

    if event.event_type == "search_query":
        for q in event.metadata.get("queries", []):
            extra += f'<div class="query-tag">🔎 {_esc(q)}</div>'
    elif event.event_type == "search_complete":
        ds = event.metadata.get("data_size", "?")
        extra += f'<div style="padding-left:38px;margin-top:4px;"><span class="data-badge">📦 {_esc(ds)} fetched</span></div>'
    elif event.event_type == "researcher_start":
        topic = event.metadata.get("topic", "")
        if topic:
            detail = f"Investigating: {_esc(topic[:150])}"
    elif event.event_type == "researcher_done":
        chars = event.metadata.get("chars", 0)
        extra += f'<div style="padding-left:38px;margin-top:4px;"><span class="data-badge">📄 {chars:,} chars compiled</span></div>'
    elif event.event_type == "tool_calls":
        tools = event.metadata.get("tools", [])
        ts_str = ", ".join(f"<code>{_esc(t)}</code>" for t in tools)
        detail = f"Calling: {ts_str}"
    elif event.event_type == "source_url":
        url = event.metadata.get("url", event.detail)
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) > 2:
            breadcrumb = f"{domain} › {path_parts[0]} › {path_parts[1]}…"
        elif path_parts:
            breadcrumb = f"{domain} › {' › '.join(path_parts[:2])}"
        else:
            breadcrumb = domain
        favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
        detail = ""
        extra += (
            f'<div class="source-chip">'
            f'<img src="{_esc(favicon)}" alt="" onerror="this.style.display=\'none\'">'
            f'<span class="url-breadcrumb"><a href="{_esc(url)}" target="_blank">{_esc(breadcrumb)}</a></span></div>'
        )

    return (
        f'<div class="action-card {cc}">'
        f'<div class="card-header"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-timestamp">{ts}</div></div>'
        f'<div class="card-detail">{detail}</div>'
        f'{extra}</div>'
    )


def render_activity_feed(events, show_shimmer=False):
    count = len(events)
    html = _get_inline_css()
    html += f'<div class="feed-title"><h3>⚡ Active Intelligence Feed</h3><span class="count-badge">{count} events</span></div>'

    if not events and not show_shimmer:
        html += '<div class="empty-state"><div class="icon">🔬</div><div class="msg">Submit a research query to begin the intelligence pipeline</div></div>'
        st.html(html)
        return

    for event in events:
        html += _render_action_card(event)

    if show_shimmer:
        html += '<div class="shimmer-card"><div class="shimmer-line short"></div><div class="shimmer-line long"></div><div class="shimmer-line medium"></div></div>'

    st.html(html)

    # Expandable thinking blocks via native Streamlit
    for event in events:
        if event.event_type == "thinking" and len(event.detail) > 120:
            with st.expander(f"🧠 View Full Chain — {event.detail[:60]}...", expanded=False):
                st.write(event.detail)


def render_source_cards(raw_trace_text):
    urls = extract_urls_from_text(raw_trace_text)
    if not urls:
        return
    html = _get_inline_css()
    html += f'<div class="feed-title" style="margin-top:16px;"><h3>🌐 Sources Discovered</h3><span class="count-badge">{len(urls)} sources</span></div>'
    for u in urls[:20]:
        html += (
            f'<div class="source-chip">'
            f'<img src="{_esc(u["favicon"])}" alt="" onerror="this.style.display=\'none\'">'
            f'<span class="url-breadcrumb"><a href="{_esc(u["url"])}" target="_blank">{_esc(u["breadcrumb"])}</a></span></div>'
        )
    st.html(html)


# ── Knowledge Sidebar ──────────────────────────────────────────────────────────

def render_knowledge_sidebar(events, raw_trace=""):
    rs = [e for e in events if e.event_type == "researcher_start"]
    rd = [e for e in events if e.event_type == "researcher_done"]
    se = [e for e in events if e.event_type == "search_query"]
    td = sum(e.metadata.get("data_bytes", 0) for e in events if e.event_type == "search_complete")
    urls = extract_urls_from_text(raw_trace)

    html = _get_inline_css()
    html += '<div class="sidebar-section"><div class="sidebar-section-title">🧬 Knowledge Graph</div>'
    html += f'<div class="stat-row"><span class="stat-label">Researchers Active</span><span class="stat-value accent-blue">{len(rs)}</span></div>'
    html += f'<div class="stat-row"><span class="stat-label">Research Complete</span><span class="stat-value accent-emerald">{len(rd)}</span></div>'
    html += f'<div class="stat-row"><span class="stat-label">Search Rounds</span><span class="stat-value accent-amber">{len(se)}</span></div>'
    html += f'<div class="stat-row"><span class="stat-label">Sites Scanned</span><span class="stat-value accent-blue">{len(urls)}</span></div>'
    html += f'<div class="stat-row"><span class="stat-label">Data Fetched</span><span class="stat-value accent-emerald">{_fmt(td)}</span></div>'
    html += '</div>'
    st.html(html)

    if rs:
        done_topics = {e.detail for e in rd}
        qh = _get_inline_css()
        qh += '<div class="sidebar-section"><div class="sidebar-section-title">🔬 Active Queries</div>'
        for r in rs:
            topic = r.metadata.get("topic", r.detail)[:80]
            is_done = any(topic[:30] in d for d in done_topics) or len(rd) >= len(rs)
            icon = "✅" if is_done else "⏳"
            qh += f'<div class="query-item"><span class="query-status">{icon}</span><span class="query-text">{_esc(topic)}</span></div>'
        qh += '</div>'
        st.html(qh)


def _fmt(chars):
    if chars >= 1024 * 1024:
        return f"{chars / (1024*1024):.1f} MB"
    elif chars >= 1024:
        return f"{chars / 1024:.1f} KB"
    elif chars > 0:
        return f"{chars} B"
    return "0 B"


# ── Final Report ───────────────────────────────────────────────────────────────

def render_final_report(report, brief="", research_id=None):
    id_text = f" (ID: #{research_id})" if research_id else ""
    html = _get_inline_css()
    html += (
        f'<div class="success-banner"><div class="icon">✅</div><div>'
        f'<div class="text">Research Complete{_esc(id_text)}</div>'
        f'<div class="subtext">All findings have been synthesized into a comprehensive report</div>'
        f'</div></div>'
    )
    st.html(html)

    if brief:
        with st.expander("📋 Research Brief", expanded=False):
            st.markdown(brief)

    # Use st.markdown for report so markdown syntax (##, **, links) renders properly
    st.markdown("---")
    st.markdown(report)


def render_methodology_accordion(events, raw_trace=""):
    with st.expander("📊 Research Methodology — View Intelligence Trace", expanded=False):
        researchers = len([e for e in events if e.event_type == "researcher_start"])
        searches = len([e for e in events if e.event_type == "search_query"])
        thinking = len([e for e in events if e.event_type == "thinking"])
        urls = extract_urls_from_text(raw_trace)
        cols = st.columns(4)
        cols[0].metric("Researchers", researchers)
        cols[1].metric("Search Rounds", searches)
        cols[2].metric("Reasoning Steps", thinking)
        cols[3].metric("Sources Found", len(urls))
        if raw_trace:
            st.code(raw_trace, language="text")


def inject_auto_scroll():
    st.html(
        '<script>'
        'const mc=window.parent.document.querySelector("section.main");'
        'if(mc){mc.scrollTop=mc.scrollHeight;}'
        '</script>'
    )
