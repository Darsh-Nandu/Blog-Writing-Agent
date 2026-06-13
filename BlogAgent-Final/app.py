"""
BWA 4.0 - Blog Writing Agent
Streamlit UI with:
  • Ollama + Groq provider support
  • Human-in-the-loop checkpoints (router & plan approval)
  • Live step-by-step progress feed
  • Markdown preview & download
"""

from __future__ import annotations
import os, time, json, re
from datetime import date
from pathlib import Path


# helper: very simple markdown → HTML (just for headings/bold in preview)
def _md_to_html(md: str) -> str:
    """Minimal markdown → HTML for the preview card (Streamlit handles most of it)."""
    lines = []
    in_code = False
    for line in md.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            lines.append("<pre><code>" if in_code else "</code></pre>")
            continue
        if in_code:
            lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue
        line = re.sub(r"^# (.+)$",  r"<h1>\1</h1>", line)
        line = re.sub(r"^## (.+)$", r"<h2>\1</h2>", line)
        line = re.sub(r"^### (.+)$",r"<h3>\1</h3>", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", line)
        line = re.sub(r"`(.+?)`",       r"<code>\1</code>", line)
        line = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', line)
        if line.startswith("- ") or line.startswith("* "):
            line = "<li>" + line[2:] + "</li>"
        elif line.strip() == "":
            line = "<br>"
        else:
            line = "<p>" + line + "</p>"
        lines.append(line)
    return "\n".join(lines)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# page config (must be first Streamlit call) 
st.set_page_config(
    page_title="BWA 4.0 — Blog Writing Agent",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# local imports (after st.set_page_config)
from llm_factory import GROQ_MODELS, OLLAMA_MODELS, get_llm
from custom_objects import Plan, Task, EvidenceItem
from nodes import router_node, research_node, orchestrator_node, worker_node, reducer_node

# global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── app shell ── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
[data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 600px at 10% -10%, #1b1f2e 0%, #0c0e14 45%, #0a0b10 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14161e 0%, #0f1117 100%);
    border-right: 1px solid #23262f;
}
[data-testid="stSidebar"] *        { color: #cbd2dc !important; }
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label { font-weight: 500; }

/* material symbols rendered inline */
[data-testid="stIconMaterial"] { vertical-align: -4px; }

/* ── header ── */
.bwa-header {
    display: flex; align-items: center; gap: 16px;
    padding: 22px 4px 22px;
    border-bottom: 1px solid #23262f;
    margin-bottom: 28px;
}
.bwa-logo {
    width: 46px; height: 46px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6 55%, #ec4899);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 24px -8px rgba(99,102,241,.6);
    flex-shrink: 0;
}
.bwa-logo svg { width: 24px; height: 24px; stroke: #fff; }
.bwa-header h1 {
    font-size: 1.6rem; font-weight: 800; color: #f8fafc; margin: 0;
    letter-spacing: -0.02em;
}
.bwa-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; font-size: 0.62rem; font-weight: 700;
    padding: 3px 9px; border-radius: 99px; letter-spacing: .08em;
    box-shadow: 0 2px 8px -2px rgba(139,92,246,.6);
}

/* ── cards ── */
.card {
    background: linear-gradient(180deg, #1c1f2b 0%, #181b25 100%);
    border: 1px solid #262a36;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 1px 0 rgba(255,255,255,.02) inset, 0 12px 24px -16px rgba(0,0,0,.5);
}
.card-title {
    font-size: 0.74rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #7c8493; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}

/* ── step log ── */
.step-log {
    background: #0e1016;
    border: 1px solid #262a36;
    border-radius: 12px;
    padding: 16px 20px;
    max-height: 340px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    color: #cbd2dc;
    line-height: 1.7;
}
.step-log::-webkit-scrollbar { width: 8px; }
.step-log::-webkit-scrollbar-thumb { background: #2a2e3b; border-radius: 99px; }
.log-line { margin: 2px 0; display: flex; align-items: center; gap: 6px; }
.log-ts   { color: #4b5563; margin-right: 4px; }

/* ── HITL panels ── */
.hitl-box {
    background: linear-gradient(135deg, rgba(245,158,11,.10), rgba(245,158,11,.02));
    border: 1px solid rgba(245,158,11,.35);
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 18px;
}
.hitl-title {
    color: #fbbf24; font-weight: 700; font-size: 1rem; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}
.hitl-sub { color: #a1a8b5; font-size: 0.86rem; margin-bottom: 0; }

/* ── router chip ── */
.mode-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 13px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 700; margin-right: 8px;
}
.chip-open   { background: rgba(59,130,246,.16); color: #93c5fd; border: 1px solid rgba(59,130,246,.3); }
.chip-hybrid { background: rgba(16,185,129,.16); color: #6ee7b7; border: 1px solid rgba(16,185,129,.3); }
.chip-closed { background: rgba(148,163,184,.14); color: #cbd5e1; border: 1px solid rgba(148,163,184,.25); }

/* ── task cards ── */
.task-card {
    background: #11141c;
    border: 1px solid #262a36;
    border-left: 3px solid #6366f1;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: border-color .15s ease;
}
.task-card:hover { border-left-color: #8b5cf6; }
.task-title { color: #f8fafc; font-weight: 600; font-size: 0.93rem; }
.task-meta  { color: #7c8493; font-size: 0.75rem; margin-top: 4px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.task-tag   { background: #1f2433; color: #9ca3af; font-size: 0.7rem; padding: 2px 9px; border-radius: 99px; }
.task-flag  { color: #a78bfa; font-size: 0.75rem; display: inline-flex; align-items: center; gap: 4px; }
.task-bullet { color: #a1a8b5; font-size: 0.83rem; margin-top: 8px; list-style: disc; padding-left: 18px; }

/* ── progress bar ── */
.prog-wrap { background: #1a1d27; border-radius: 99px; height: 8px; margin: 8px 0; }
.prog-fill {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    height: 8px; border-radius: 99px;
    transition: width .4s ease;
}

/* ── final blog ── */
.blog-preview {
    background: linear-gradient(180deg, #1c1f2b 0%, #181b25 100%);
    border: 1px solid #262a36;
    border-radius: 16px;
    padding: 36px 44px;
    color: #dde1e8;
    line-height: 1.8;
    font-size: 0.96rem;
    box-shadow: 0 20px 40px -28px rgba(0,0,0,.6);
}
.blog-preview h1 { color: #f8fafc; font-size: 2rem; font-weight: 800; border-bottom: 1px solid #262a36; padding-bottom: 14px; letter-spacing: -0.02em; }
.blog-preview h2 { color: #eef0f4; font-size: 1.35rem; font-weight: 700; margin-top: 30px; }
.blog-preview h3 { color: #e5e7eb; font-size: 1.1rem; font-weight: 700; margin-top: 22px; }
.blog-preview code { background: #0e1016; padding: 2px 6px; border-radius: 4px; color: #a78bfa; font-size: 0.87em; font-family: 'JetBrains Mono', monospace; }
.blog-preview pre  { background: #0e1016; border: 1px solid #262a36; border-radius: 10px; padding: 18px; overflow-x: auto; }
.blog-preview a    { color: #818cf8; text-decoration: none; border-bottom: 1px solid rgba(129,140,248,.35); }

/* ── score badge ── */
.score-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 54px; height: 54px; border-radius: 50%;
    font-size: 1.3rem; font-weight: 800; color: white;
}
.score-hi  { background: linear-gradient(135deg, #10b981, #059669); }
.score-med { background: linear-gradient(135deg, #f59e0b, #d97706); }
.score-lo  { background: linear-gradient(135deg, #ef4444, #dc2626); }

/* ── buttons ── */
div[data-testid="stButton"] > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    border: 1px solid #2a2e3b !important;
    transition: all .15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #6366f1 !important;
    color: #c7d2fe !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    box-shadow: 0 8px 20px -10px rgba(139,92,246,.7) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    filter: brightness(1.08);
    color: #fff !important;
}
div[data-testid="stDownloadButton"] > button {
    border-radius: 9px !important; font-weight: 600 !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    box-shadow: 0 8px 20px -10px rgba(139,92,246,.7) !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #262a36; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px 8px 0 0; font-weight: 600;
    color: #7c8493; padding: 10px 18px;
}
.stTabs [aria-selected="true"] { color: #c7d2fe !important; background: #1c1f2b; }

/* ── metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #1c1f2b 0%, #181b25 100%);
    border: 1px solid #262a36; border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #7c8493 !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800 !important; }

/* ── misc ── */
[data-testid="stMarkdownContainer"] p { color: #cbd2dc; }
.stAlert { border-radius: 12px !important; border: 1px solid #262a36 !important; }
.stExpander { border: 1px solid #262a36 !important; border-radius: 12px !important; background: #14161e; }
hr { border-color: #262a36 !important; }
</style>
""", unsafe_allow_html=True)

# ── session state init ────────────────────────────────────────────────────────
def _init():
    defaults = {
        "stage":          "idle",    # idle | routing | hitl_router | researching | orchestrating | hitl_plan | writing | done
        "log_lines":      [],
        "router_data":    None,      # {mode, queries, needs_research, reason}
        "evidence":       [],
        "plan":           None,
        "sections":       [],
        "final_blog":     "",
        "topic":          "",
        "provider":       "groq",
        "model_name":     "llama-3.3-70b-versatile",
        "as_of":          date.today().isoformat(),
        "recency_days":   7,
        "mode":           "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# log helper
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_lines.append(f'<span class="log-ts">[{ts}]</span> {msg}')


# sidebar
with st.sidebar:
    st.markdown("##### :material/tune: Configuration")
    st.divider()

    # Provider
    st.markdown("**LLM Provider**")
    provider = st.radio(
        "provider", ["groq", "ollama"],
        format_func=lambda x: "Groq · Cloud" if x == "groq" else "Ollama · Local",
        index=0 if st.session_state.provider == "groq" else 1,
        label_visibility="collapsed",
    )
    st.session_state.provider = provider

    # Model
    st.markdown("**Model**")
    if provider == "groq":
        model_opts = GROQ_MODELS
    else:
        model_opts = OLLAMA_MODELS

    model_key = st.selectbox(
        "model", list(model_opts.keys()),
        format_func=lambda k: model_opts[k],
        label_visibility="collapsed",
    )
    st.session_state.model_name = model_key

    # API keys
    st.divider()
    st.markdown("**API Keys**")

    if provider == "groq":
        groq_key = st.text_input(
            "Groq API Key", type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get one free at console.groq.com",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

    tavily_key = st.text_input(
        "Tavily API Key", type="password",
        value=os.getenv("TAVILY_API_KEY", ""),
        help="Required for web research. Free at tavily.com",
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.markdown("**Generation Settings**")
    _ = st.slider("Max evidence items", 3, 20, 10, help="How many web results to retain per run")

    st.divider()
    st.caption("BWA 4.0 · Human-in-the-Loop Edition")
    st.caption("Supports Groq + Ollama · Web research via Tavily")

    if st.button("Reset Session", icon=":material/refresh:", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init()
        st.rerun()


# main content
st.markdown("""
<div class="bwa-header">
  <div class="bwa-logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 20h9"/>
      <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z"/>
    </svg>
  </div>
  <div>
    <div style="display:flex;align-items:center;gap:10px">
      <h1>Blog Writing Agent</h1>
      <span class="bwa-badge">v4.0</span>
    </div>
    <p style="color:#7c8493;margin:4px 0 0;font-size:0.9rem">
      AI-powered blog generator · Human-in-the-loop · Groq + Ollama
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


# STAGE: IDLE - topic input
if st.session_state.stage == "idle":
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="card"><div class="card-title">:material/edit_note: Blog Topic</div>', unsafe_allow_html=True)
        topic = st.text_area(
            "What should the blog be about?",
            placeholder="e.g.  'How LangGraph enables stateful multi-agent workflows'\n"
                        "      'State of open-source LLMs in 2026'\n"
                        "      'Building a RAG system with Chroma and Ollama'",
            height=110,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card"><div class="card-title">:material/lightbulb: Quick Tips</div>', unsafe_allow_html=True)
        st.markdown("""
        <ul style="color:#a1a8b5;font-size:0.83rem;padding-left:16px;margin:0">
          <li>Be specific for better results</li>
          <li>Include audience hints if needed</li>
          <li>Time-sensitive topics → open_book mode</li>
          <li>Fundamentals → closed_book (no web)</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Generate Blog", icon=":material/rocket_launch:", type="primary", use_container_width=True, disabled=not topic.strip()):
            st.session_state.topic    = topic.strip()
            st.session_state.stage   = "routing"
            st.session_state.log_lines = []
            st.rerun()


# STAGE: ROUTING
elif st.session_state.stage == "routing":
    st.info(f"Analysing topic: **{st.session_state.topic}**", icon=":material/satellite_alt:")

    with st.spinner("Running router…"):
        fake_state = {
            "topic":      st.session_state.topic,
            "provider":   st.session_state.provider,
            "model_name": st.session_state.model_name,
            "as_of":      st.session_state.as_of,
        }
        try:
            result = router_node(fake_state, log=log)
            st.session_state.router_data = result
            st.session_state.mode          = result["mode"]
            st.session_state.recency_days  = result["recency_days"]
            st.session_state.stage         = "hitl_router"
            st.rerun()
        except Exception as e:
            st.error(f"Router error: {e}")
            st.session_state.stage = "idle"


# STAGE: HITL - router approval
elif st.session_state.stage == "hitl_router":
    rd = st.session_state.router_data
    mode = rd["mode"]

    # Log display
    if st.session_state.log_lines:
        st.markdown('<div class="step-log">' + "<br>".join(st.session_state.log_lines) + '</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # HITL box
    chip_cls = {"open_book": "chip-open", "hybrid": "chip-hybrid", "closed_book": "chip-closed"}[mode]
    chip_lbl = {
        "open_book": ':material/public: Open Book',
        "hybrid": ':material/shuffle: Hybrid',
        "closed_book": ':material/menu_book: Closed Book',
    }[mode]

    st.markdown(f"""
    <div class="hitl-box">
      <div class="hitl-title">:material/flag: Checkpoint 1 of 2 — Review Research Strategy</div>
      <div class="hitl-sub">The router has analysed your topic. Please confirm or redirect before research begins.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'**Mode chosen:** <span class="mode-chip {chip_cls}">{chip_lbl}</span>', unsafe_allow_html=True)
        st.markdown(f"**Research needed:** {':material/check_circle: Yes' if rd['needs_research'] else ':material/cancel: No'}")
        st.markdown(f"**Recency window:** {rd['recency_days']} days")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if rd.get("queries"):
            st.markdown('<div class="card"><div class="card-title">:material/travel_explore: Planned Search Queries</div>', unsafe_allow_html=True)
            for q in rd["queries"]:
                st.markdown(f"`{q}`")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><p style="color:#7c8493">No web research planned (closed_book mode).</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    feedback = st.text_area(
        "Optional feedback / instructions for the next steps",
        placeholder="e.g. 'Focus more on production use cases' · 'Add a query about LangGraph v0.4' · 'Use a more beginner-friendly tone'",
        height=80,
    )

    col_ok, col_back = st.columns([1, 4])
    with col_ok:
        if st.button("Approve & Continue", icon=":material/check_circle:", type="primary", use_container_width=True):
            st.session_state.router_feedback = feedback
            st.session_state.stage = "researching" if rd["needs_research"] else "orchestrating"
            st.rerun()
    with col_back:
        if st.button("Back to Topic", icon=":material/arrow_back:", use_container_width=True):
            st.session_state.stage = "idle"
            st.rerun()


# STAGE: RESEARCHING
elif st.session_state.stage == "researching":
    st.info("Researching the web with Tavily…", icon=":material/travel_explore:")

    log_placeholder = st.empty()

    with st.spinner("Fetching & filtering evidence…"):
        fake_state = {
            "topic":        st.session_state.topic,
            "provider":     st.session_state.provider,
            "model_name":   st.session_state.model_name,
            "as_of":        st.session_state.as_of,
            "mode":         st.session_state.mode,
            "queries":      st.session_state.router_data["queries"],
            "recency_days": st.session_state.recency_days,
        }
        try:
            result = research_node(fake_state, log=log)
            st.session_state.evidence = result["evidence"]
            st.session_state.stage    = "orchestrating"

            log_placeholder.markdown(
                '<div class="step-log">' + "<br>".join(st.session_state.log_lines) + '</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.3)
            st.rerun()
        except Exception as e:
            st.error(f"Research error: {e}")
            st.session_state.stage = "hitl_router"


# STAGE: ORCHESTRATING
elif st.session_state.stage == "orchestrating":
    st.info("Generating blog outline…", icon=":material/route:")

    with st.spinner("Orchestrating plan…"):
        fake_state = {
            "topic":        st.session_state.topic,
            "provider":     st.session_state.provider,
            "model_name":   st.session_state.model_name,
            "as_of":        st.session_state.as_of,
            "mode":         st.session_state.mode,
            "evidence":     st.session_state.evidence,
            "recency_days": st.session_state.recency_days,
        }
        try:
            result = orchestrator_node(fake_state, log=log)
            st.session_state.plan  = result["plan"]
            st.session_state.stage = "hitl_plan"
            st.rerun()
        except Exception as e:
            st.error(f"Orchestrator error: {e}")
            st.session_state.stage = "hitl_router"


# STAGE: HITL - plan approval
elif st.session_state.stage == "hitl_plan":
    plan: Plan = st.session_state.plan

    # Log
    if st.session_state.log_lines:
        st.markdown('<div class="step-log">' + "<br>".join(st.session_state.log_lines) + '</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hitl-box">
      <div class="hitl-title">:material/flag: Checkpoint 2 of 2 — Review Blog Outline</div>
      <div class="hitl-sub">The orchestrator has generated a plan. Review each section before writing begins.</div>
    </div>
    """, unsafe_allow_html=True)

    # Plan metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Blog Kind", plan.blog_kind.replace("_", " ").title())
    with col2:
        st.metric("Sections", len(plan.tasks))
    with col3:
        total_words = sum(t.target_words for t in plan.tasks)
        st.metric("Est. Total Words", f"~{total_words:,}")

    st.markdown(f"**Title:** {plan.blog_title}")
    st.markdown(f"**Audience:** {plan.audience} · **Tone:** {plan.tone}")

    # Evidence summary
    if st.session_state.evidence:
        with st.expander(f"Research Evidence ({len(st.session_state.evidence)} items)", icon=":material/menu_book:", expanded=False):
            for e in st.session_state.evidence[:12]:
                st.markdown(f"- [{e.title or e.url}]({e.url}) — *{e.published_at or 'date unknown'}*")

    # Section cards
    st.markdown("**Planned Sections:**")
    for task in plan.tasks:
        tags_html = "".join(
            f'<span class="task-tag">{t}</span>'
            for t in task.tags
        )
        flags = []
        if task.requires_code:      flags.append(':material/code: code')
        if task.requires_citations: flags.append(':material/attach_file: citations')
        if task.requires_research:  flags.append(':material/travel_explore: research')
        flags_html = "".join(f'<span class="task-flag">{f}</span>' for f in flags)

        bullets_html = "".join(f"<li>{b}</li>" for b in task.bullets)
        st.markdown(f"""
        <div class="task-card">
          <div class="task-title">§{task.id} · {task.title}</div>
          <div class="task-meta">~{task.target_words} words &nbsp;·&nbsp; {tags_html} &nbsp;{flags_html}</div>
          <ul class="task-bullet">{bullets_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    feedback = st.text_area(
        "Optional feedback before writing",
        placeholder="e.g. 'Add a section on testing' · 'Make section 3 more code-heavy' · 'Swap sections 2 and 4'",
        height=80,
    )

    col_ok, col_regen, col_back = st.columns([1, 1, 3])
    with col_ok:
        if st.button("Approve & Write", icon=":material/check_circle:", type="primary", use_container_width=True):
            st.session_state.plan_feedback = feedback
            st.session_state.stage = "writing"
            st.rerun()
    with col_regen:
        if st.button("Regenerate Plan", icon=":material/refresh:", use_container_width=True):
            st.session_state.stage = "orchestrating"
            st.rerun()
    with col_back:
        if st.button("Back to Router Review", icon=":material/arrow_back:", use_container_width=True):
            st.session_state.stage = "hitl_router"
            st.rerun()


# STAGE: WRITING
elif st.session_state.stage == "writing":
    plan: Plan = st.session_state.plan
    n = len(plan.tasks)

    st.markdown(f"### :material/edit: Writing **{plan.blog_title}**")

    progress_bar = st.progress(0, text="Starting…")
    log_box      = st.empty()
    section_feed = st.empty()

    sections = []
    for i, task in enumerate(plan.tasks):
        pct = int((i / n) * 100)
        progress_bar.progress(pct, text=f"Writing section {i+1}/{n}: {task.title}")

        log(f":material/edit_note: Section {task.id}/{n}: **{task.title}**")
        log_box.markdown(
            '<div class="step-log">' + "<br>".join(st.session_state.log_lines[-12:]) + '</div>',
            unsafe_allow_html=True
        )

        payload = {
            "task":         task.model_dump(),
            "topic":        st.session_state.topic,
            "mode":         st.session_state.mode,
            "plan":         plan.model_dump(),
            "evidence":     [e.model_dump() for e in st.session_state.evidence],
            "as_of":        st.session_state.as_of,
            "recency_days": st.session_state.recency_days,
            "provider":     st.session_state.provider,
            "model_name":   st.session_state.model_name,
        }
        try:
            result = worker_node(payload, log=log)
            sections.extend(result["sections"])

            # Live preview of written section
            _, section_md = result["sections"][0]
            with section_feed.expander(f"{task.title}", icon=":material/check_circle:", expanded=False):
                st.markdown(section_md)

        except Exception as e:
            st.error(f"Worker error on section {task.id}: {e}")

    progress_bar.progress(100, text="Assembling final blog…")
    log(":material/inventory_2: Assembling all sections…")

    # Reducer
    fake_state = {
        "plan":       plan,
        "sections":   sections,
        "provider":   st.session_state.provider,
        "model_name": st.session_state.model_name,
    }
    from nodes import reducer_node as _reduce
    result = _reduce(fake_state, log=log)
    st.session_state.final_blog = result["final"]
    st.session_state.sections   = sections
    st.session_state.stage      = "done"

    log(":material/celebration: Blog complete!")
    log_box.markdown(
        '<div class="step-log">' + "<br>".join(st.session_state.log_lines[-12:]) + '</div>',
        unsafe_allow_html=True
    )
    time.sleep(0.5)
    st.rerun()


# STAGE: DONE
elif st.session_state.stage == "done":
    plan: Plan      = st.session_state.plan
    final: str      = st.session_state.final_blog
    word_count      = len(final.split())
    section_count   = len(st.session_state.sections)

    # stats banner
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(":material/edit_note: Words",    f"{word_count:,}")
    col2.metric(":material/bookmarks: Sections", section_count)
    col3.metric(":material/public: Mode",     st.session_state.mode.replace("_", " ").title())
    col4.metric(":material/menu_book: Evidence", len(st.session_state.evidence))

    st.success(f"**{plan.blog_title}** — generation complete!", icon=":material/check_circle:")

    # tab layout
    tab_preview, tab_raw, tab_log = st.tabs([":material/book_4: Preview", ":material/description: Raw Markdown", ":material/terminal: Generation Log"])

    with tab_preview:
        st.markdown(f'<div class="blog-preview">{_md_to_html(final)}</div>', unsafe_allow_html=True)

    with tab_raw:
        st.code(final, language="markdown", line_numbers=True)

    with tab_log:
        st.markdown(
            '<div class="step-log">' + "<br>".join(st.session_state.log_lines) + '</div>',
            unsafe_allow_html=True
        )

    # ── actions ──
    st.markdown("---")
    col_dl, col_cp, col_new = st.columns([1, 1, 2])

    with col_dl:
        filename = plan.blog_title.replace(" ", "_").replace("/", "-")[:60] + ".md"
        st.download_button(
            "Download .md",
            icon=":material/download:",
            data=final.encode(),
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )

    with col_cp:
        if st.button("Copy to Clipboard", icon=":material/content_copy:", use_container_width=True):
            st.write("""<script>
            navigator.clipboard.writeText(document.querySelector('code').innerText);
            </script>""", unsafe_allow_html=True)
            st.toast("Copied!")

    with col_new:
        if st.button("Write Another Blog", icon=":material/auto_awesome:", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init()
            st.rerun()