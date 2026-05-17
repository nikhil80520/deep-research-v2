"""
Dashboard CSS styles — injected into Streamlit via st.markdown(unsafe_allow_html=True).
Dark-themed glassmorphism design with pulse/shimmer animations.
"""

GOOGLE_FONTS_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

DASHBOARD_CSS = """
<style>
/* ── Global Typography ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Color Palette (CSS Variables) ─────────────────────────────────── */
:root {
    --bg-primary: #0a0e1a;
    --bg-card: rgba(15, 23, 42, 0.75);
    --bg-card-hover: rgba(20, 30, 55, 0.85);
    --bg-glass: rgba(255, 255, 255, 0.03);
    --border-subtle: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.4);
    --accent-blue: #6366f1;
    --accent-blue-light: #818cf8;
    --accent-emerald: #10b981;
    --accent-emerald-light: #34d399;
    --accent-amber: #f59e0b;
    --accent-amber-light: #fbbf24;
    --accent-rose: #f43f5e;
    --accent-purple: #a855f7;
    --accent-purple-light: #c084fc;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.1);
}

/* ── Command Center Header ─────────────────────────────────────────── */
.command-center {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 27, 75, 0.7) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-glow);
    position: relative;
    overflow: hidden;
}

.command-center::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
    opacity: 0.6;
}

.command-center-title {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
}

.command-center-title h1 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #f1f5f9, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}

.command-center-title .subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ── Pulse Badge ───────────────────────────────────────────────────── */
.pulse-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.pulse-badge .pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.pulse-badge.idle {
    background: rgba(16, 185, 129, 0.12);
    color: var(--accent-emerald-light);
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.pulse-badge.idle .pulse-dot {
    background: var(--accent-emerald);
}

.pulse-badge.researching {
    background: rgba(99, 102, 241, 0.12);
    color: var(--accent-blue-light);
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.pulse-badge.researching .pulse-dot {
    background: var(--accent-blue);
    animation: pulse-anim 1.5s ease-in-out infinite;
}

.pulse-badge.synthesizing {
    background: rgba(245, 158, 11, 0.12);
    color: var(--accent-amber-light);
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.pulse-badge.synthesizing .pulse-dot {
    background: var(--accent-amber);
    animation: pulse-anim 1.2s ease-in-out infinite;
}

.pulse-badge.complete {
    background: rgba(16, 185, 129, 0.12);
    color: var(--accent-emerald-light);
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.pulse-badge.complete .pulse-dot {
    background: var(--accent-emerald);
}

@keyframes pulse-anim {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 currentColor; }
    50% { opacity: 0.6; transform: scale(1.6); box-shadow: 0 0 8px 2px currentColor; }
}

/* ── Pipeline Breadcrumb ───────────────────────────────────────────── */
.pipeline-breadcrumb {
    display: flex;
    align-items: center;
    gap: 0;
    margin-top: 4px;
    flex-wrap: wrap;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 6px;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
    transition: all 0.3s ease;
    position: relative;
}

.pipeline-step .step-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid var(--text-muted);
    background: transparent;
    transition: all 0.3s ease;
    flex-shrink: 0;
}

.pipeline-step.active {
    color: var(--accent-blue-light);
}
.pipeline-step.active .step-dot {
    border-color: var(--accent-blue);
    background: var(--accent-blue);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    animation: pulse-anim 2s ease-in-out infinite;
}

.pipeline-step.completed {
    color: var(--accent-emerald-light);
}
.pipeline-step.completed .step-dot {
    border-color: var(--accent-emerald);
    background: var(--accent-emerald);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.3);
}

.pipeline-arrow {
    color: var(--text-muted);
    font-size: 0.7rem;
    margin: 0 2px;
    opacity: 0.5;
}

.pipeline-arrow.active {
    color: var(--accent-blue-light);
    opacity: 1;
}

/* ── Action Cards ──────────────────────────────────────────────────── */
.action-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    backdrop-filter: blur(12px);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.action-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}

.action-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
}

/* Card type borders */
.action-card.thinking::before { background: linear-gradient(180deg, var(--accent-purple), var(--accent-purple-light)); }
.action-card.search::before { background: linear-gradient(180deg, var(--accent-blue), var(--accent-blue-light)); }
.action-card.researcher::before { background: linear-gradient(180deg, var(--accent-emerald), var(--accent-emerald-light)); }
.action-card.error::before { background: linear-gradient(180deg, var(--accent-rose), #fb7185); }
.action-card.info::before { background: linear-gradient(180deg, var(--text-muted), var(--text-secondary)); }
.action-card.tool::before { background: linear-gradient(180deg, var(--accent-amber), var(--accent-amber-light)); }

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.card-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: var(--bg-glass);
}

.card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
}

.card-timestamp {
    font-size: 0.68rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}

.card-detail {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
    padding-left: 38px;
}

.card-detail code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    background: rgba(99, 102, 241, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--accent-blue-light);
}

/* ── Source Chip ────────────────────────────────────────────────────── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 8px;
    margin: 4px 4px 4px 38px;
    transition: all 0.2s ease;
    max-width: 100%;
}

.source-chip:hover {
    background: rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.3);
}

.source-chip img {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    flex-shrink: 0;
}

.source-chip .url-breadcrumb {
    font-size: 0.72rem;
    color: var(--accent-blue-light);
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-decoration: none;
}

.source-chip .url-breadcrumb a {
    color: var(--accent-blue-light);
    text-decoration: none;
}
.source-chip .url-breadcrumb a:hover {
    text-decoration: underline;
}

.data-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    background: rgba(16, 185, 129, 0.12);
    border-radius: 100px;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--accent-emerald-light);
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

/* ── Query Tags ────────────────────────────────────────────────────── */
.query-tag {
    display: inline-block;
    padding: 4px 10px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 6px;
    font-size: 0.73rem;
    color: var(--accent-blue-light);
    font-family: 'JetBrains Mono', monospace;
    margin: 3px 2px 3px 38px;
}

/* ── Shimmer / Skeleton Loader ─────────────────────────────────────── */
.shimmer-card {
    background: linear-gradient(90deg,
        var(--bg-card) 25%,
        rgba(99, 102, 241, 0.08) 50%,
        var(--bg-card) 75%
    );
    background-size: 200% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border: 1px solid var(--border-subtle);
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.shimmer-line {
    height: 12px;
    background: rgba(99, 102, 241, 0.06);
    border-radius: 6px;
    margin-bottom: 8px;
}
.shimmer-line.short { width: 40%; }
.shimmer-line.medium { width: 65%; }
.shimmer-line.long { width: 85%; }

/* ── Processing Animation ──────────────────────────────────────────── */
.processing-overlay {
    position: relative;
}
.processing-overlay::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(99, 102, 241, 0.05) 50%,
        transparent 100%
    );
    background-size: 200% 100%;
    animation: shimmer 2s ease-in-out infinite;
    border-radius: 12px;
    pointer-events: none;
}

/* ── Knowledge Sidebar ─────────────────────────────────────────────── */
.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    backdrop-filter: blur(12px);
}

.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.06);
}

.stat-row:last-child {
    border-bottom: none;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.stat-value {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}

.stat-value.accent-blue { color: var(--accent-blue-light); }
.stat-value.accent-emerald { color: var(--accent-emerald-light); }
.stat-value.accent-amber { color: var(--accent-amber-light); }

.query-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.06);
}

.query-item:last-child {
    border-bottom: none;
}

.query-status {
    flex-shrink: 0;
    font-size: 0.8rem;
    margin-top: 1px;
}

.query-text {
    font-size: 0.73rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* ── Final Report Styling ──────────────────────────────────────────── */
.final-report-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(30, 27, 75, 0.3));
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 32px 40px;
    margin-top: 16px;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-glow);
    position: relative;
    overflow: hidden;
}

.final-report-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-emerald), var(--accent-blue), var(--accent-purple));
}

.final-report-container h1,
.final-report-container h2,
.final-report-container h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

.final-report-container h1 { font-size: 1.5rem !important; margin-top: 0 !important; }
.final-report-container h2 { font-size: 1.2rem !important; margin-top: 24px !important; }
.final-report-container h3 { font-size: 1rem !important; margin-top: 20px !important; }

.final-report-container p, .final-report-container li {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

.final-report-container a {
    color: var(--accent-blue-light);
    text-decoration: none;
    border-bottom: 1px solid rgba(99, 102, 241, 0.3);
}
.final-report-container a:hover {
    border-bottom-color: var(--accent-blue-light);
}

.final-report-container code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(99, 102, 241, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
}

/* ── Methodology Accordion ─────────────────────────────────────────── */
.methodology-header {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
    margin-bottom: 16px;
}

.methodology-header:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
}

/* ── Activity Feed Container ───────────────────────────────────────── */
.activity-feed {
    max-height: 600px;
    overflow-y: auto;
    padding-right: 8px;
    scroll-behavior: smooth;
}

.activity-feed::-webkit-scrollbar {
    width: 4px;
}
.activity-feed::-webkit-scrollbar-track {
    background: transparent;
}
.activity-feed::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.2);
    border-radius: 4px;
}
.activity-feed::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.4);
}

/* ── Success Banner ────────────────────────────────────────────────── */
.success-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 12px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.success-banner .icon {
    font-size: 1.3rem;
}

.success-banner .text {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--accent-emerald-light);
}

.success-banner .subtext {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Feed section title ────────────────────────────────────────────── */
.feed-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

.feed-title h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    padding: 0 !important;
}

.feed-title .count-badge {
    background: rgba(99, 102, 241, 0.12);
    color: var(--accent-blue-light);
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Hide default Streamlit elements ───────────────────────────────── */
header[data-testid="stHeader"] {
    background: transparent !important;
}

div[data-testid="stExpander"] details {
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-card) !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

</style>
"""

# Combined injection string
FULL_STYLE_INJECTION = GOOGLE_FONTS_LINK + DASHBOARD_CSS
