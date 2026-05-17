"""
Trace Event system — parses raw print() output from agents into typed events.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional


@dataclass
class TraceEvent:
    """A single structured event from the research pipeline."""
    timestamp: datetime
    event_type: str       # "thinking", "search_query", "search_complete",
                          # "researcher_start", "researcher_done",
                          # "tool_calls", "supervisor_error", "researcher_error",
                          # "brief_writing", "clarifying", "iteration_limit",
                          # "info"
    title: str            # Short label for the card header
    detail: str           # Full content / snippet
    metadata: dict = field(default_factory=dict)
    # metadata keys by type:
    #   thinking:         {"snippet": str}
    #   search_query:     {"queries": list[str]}
    #   search_complete:  {"data_size": str, "data_bytes": int}
    #   researcher_start: {"topic": str}
    #   researcher_done:  {"chars": int}
    #   tool_calls:       {"tools": list[str]}


# ── Regex patterns for parsing existing print() lines ──────────────────────

_PATTERNS = [
    # Supervisor thinking: 💭 Thinking: <text>...
    (
        re.compile(r"^💭\s*Thinking:\s*(.+)$"),
        "thinking",
        lambda m: {
            "title": "Strategic Reasoning",
            "detail": m.group(1).rstrip("."),
            "metadata": {"snippet": m.group(1)[:120]},
        },
    ),
    # Researcher starting: 🔍 Researcher starting: <topic>...
    (
        re.compile(r"^🔍\s*Researcher starting:\s*(.+)$"),
        "researcher_start",
        lambda m: {
            "title": "Researcher Activated",
            "detail": m.group(1).rstrip("."),
            "metadata": {"topic": m.group(1).rstrip(".")},
        },
    ),
    # Search queries: 🔍 Search queries: <q1>, <q2>...
    (
        re.compile(r"^🔍\s*Search queries:\s*(.+)$"),
        "search_query",
        lambda m: {
            "title": "Web Search",
            "detail": m.group(1),
            "metadata": {"queries": [q.strip() for q in m.group(1).split(",")]},
        },
    ),
    # Search complete: ✅ Search complete (<N> chars)
    (
        re.compile(r"^✅\s*Search complete\s*\((\d+)\s*chars?\)"),
        "search_complete",
        lambda m: {
            "title": "Search Results Received",
            "detail": f"{int(m.group(1)):,} characters fetched",
            "metadata": {
                "data_bytes": int(m.group(1)),
                "data_size": _format_size(int(m.group(1))),
            },
        },
    ),
    # Researcher done: ✅ Researcher done: <N> chars
    (
        re.compile(r"^✅\s*Researcher done:\s*(\d+)\s*chars?"),
        "researcher_done",
        lambda m: {
            "title": "Research Complete",
            "detail": f"Compiled {int(m.group(1)):,} characters of findings",
            "metadata": {"chars": int(m.group(1))},
        },
    ),
    # Tool calls: 🔧 Researcher tool calls: <tool1>, <tool2>
    (
        re.compile(r"^🔧\s*Researcher tool calls?:\s*(.+)$"),
        "tool_calls",
        lambda m: {
            "title": "Tool Invocation",
            "detail": m.group(1),
            "metadata": {"tools": [t.strip() for t in m.group(1).split(",")]},
        },
    ),
    # Supervisor error: ⚠️  Supervisor LLM call failed: <err>
    (
        re.compile(r"^⚠️\s*Supervisor LLM call failed:\s*(.+)$"),
        "supervisor_error",
        lambda m: {
            "title": "Supervisor Error",
            "detail": m.group(1),
            "metadata": {},
        },
    ),
    # Researcher error: ⚠️  Researcher LLM call failed: <err>
    (
        re.compile(r"^⚠️\s*Researcher LLM call failed:\s*(.+)$"),
        "researcher_error",
        lambda m: {
            "title": "Researcher Error",
            "detail": m.group(1),
            "metadata": {},
        },
    ),
    # Iteration limit: ⏹️ Supervisor hit iteration limit
    (
        re.compile(r"^⏹️\s*Supervisor hit iteration limit"),
        "iteration_limit",
        lambda m: {
            "title": "Iteration Limit Reached",
            "detail": "Supervisor has reached the maximum number of iterations. Finishing research.",
            "metadata": {},
        },
    ),
    # Brief writing: 📋 Writing research brief
    (
        re.compile(r"^📋\s*Writing research brief"),
        "brief_writing",
        lambda m: {
            "title": "Writing Research Brief",
            "detail": "Transforming user query into a structured research plan.",
            "metadata": {},
        },
    ),
    # Clarifying: 🧠 Clarifying query
    (
        re.compile(r"^🧠\s*Clarifying query"),
        "clarifying",
        lambda m: {
            "title": "Analyzing Intent",
            "detail": "Determining if clarification is needed before research.",
            "metadata": {},
        },
    ),
    # Brief generation failed: ⚠️  Brief generation failed
    (
        re.compile(r"^⚠️\s*Brief generation failed:\s*(.+)$"),
        "brief_error",
        lambda m: {
            "title": "Brief Generation Error",
            "detail": m.group(1),
            "metadata": {},
        },
    ),
    # Source URL: 🌐 Source: <url>
    (
        re.compile(r"^🌐\s*Source:\s*(https?://\S+)$"),
        "source_url",
        lambda m: {
            "title": "Source Discovered",
            "detail": m.group(1),
            "metadata": {"url": m.group(1)},
        },
    ),
]


def _format_size(chars: int) -> str:
    """Format character count as human-readable size."""
    kb = chars / 1024
    if kb >= 1:
        return f"{kb:.1f}kb"
    return f"{chars}b"


def parse_trace_line(raw_line: str) -> Optional[TraceEvent]:
    """Parse a single print() line into a TraceEvent, or None if not recognized."""
    line = raw_line.strip()
    if not line:
        return None

    for pattern, event_type, extractor in _PATTERNS:
        match = pattern.match(line)
        if match:
            data = extractor(match)
            return TraceEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                title=data["title"],
                detail=data["detail"],
                metadata=data.get("metadata", {}),
            )

    # Fallback — unrecognized lines become "info" events
    # Only if they contain meaningful content (skip blank/separator lines)
    if len(line) > 3 and not all(c in "-=_" for c in line):
        return TraceEvent(
            timestamp=datetime.now(),
            event_type="info",
            title="System",
            detail=line,
            metadata={},
        )

    return None


def extract_urls_from_text(text: str) -> list[dict]:
    """Extract URLs from search result text and return parsed info."""
    url_pattern = re.compile(r"(?:URL:|🌐\s*Source:)\s*(https?://[^\s]+)")
    urls = []
    seen = set()
    for match in url_pattern.finditer(text):
        url = match.group(1).rstrip(".,;)")
        if url not in seen:
            seen.add(url)
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            # Create breadcrumb from path
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) > 2:
                breadcrumb = f"{domain} › {path_parts[0]} › {path_parts[1]}…"
            elif path_parts:
                breadcrumb = f"{domain} › {' › '.join(path_parts)}"
            else:
                breadcrumb = domain
            urls.append({
                "url": url,
                "domain": domain,
                "breadcrumb": breadcrumb,
                "favicon": f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
            })
    return urls


def determine_pipeline_stage(events: list[TraceEvent]) -> int:
    """Determine current pipeline stage (0-3) from events seen so far.

    0 = Intent Analysis
    1 = Parallel Extraction
    2 = Synthesis
    3 = Final Report
    """
    types_seen = {e.event_type for e in events}

    if any(t in types_seen for t in ("researcher_done", "iteration_limit")):
        # Check if we have all researchers done — synthesis stage
        starts = sum(1 for e in events if e.event_type == "researcher_start")
        dones = sum(1 for e in events if e.event_type == "researcher_done")
        if starts > 0 and dones >= starts:
            return 2  # Synthesis
        return 1  # Still extracting

    if any(t in types_seen for t in ("researcher_start", "search_query", "search_complete")):
        return 1  # Parallel Extraction

    if any(t in types_seen for t in ("brief_writing", "clarifying")):
        return 0  # Intent Analysis

    return 0
