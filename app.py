"""
Streamlit UI for the multi-agent research pipeline.

Run with:
    streamlit run app.py
"""

import os
import traceback
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(
    page_title="Nexus Research | Multi-Agent AI System",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C3AED, #2563EB, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }
    .badge-row { margin-bottom: 1.6rem; }
    .badge {
        display: inline-block;
        padding: 5px 14px;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(124, 58, 237, 0.4);
        background: rgba(124, 58, 237, 0.08);
        color: #C4B5FD;
    }
    .agent-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .footer-note {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "selected" not in st.session_state:
    st.session_state.selected = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_text(value) -> str:
    if value is None:
        return "_(empty response)_"
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return str(value.content)
    return str(value)


def check_env():
    likely_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY",
                   "SERPAPI_API_KEY", "GOOGLE_API_KEY"]
    present = [k for k in likely_keys if os.getenv(k)]
    return present, len(present) == 0


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Nexus Research")
    st.caption("Autonomous multi-agent research engine")
    st.markdown(
        "**Pipeline**\n"
        "1. 🔍 Search Agent\n"
        "2. 📰 Reader Agent\n"
        "3. ✍️ Writer\n"
        "4. 🧐 Critic"
    )

    present_keys, missing_all = check_env()
    st.divider()
    st.caption("Environment")
    if missing_all:
        st.warning("No API keys detected in .env")
    else:
        st.success(f"{len(present_keys)} key(s) loaded")

    st.divider()
    if st.session_state.history:
        st.markdown("**Run history**")
        for i, run in enumerate(reversed(st.session_state.history)):
            if st.button(f"📄 {run['topic'][:28]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.selected = run
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.selected = None
        st.rerun()

    st.markdown(
        "<div style='margin-top:2rem; color:#6B7280; font-size:0.75rem;'>"
        "Built with LangGraph + Streamlit</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🧠 Nexus Research</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">An autonomous multi-agent system that searches, reads, '
    'writes, and critiques research reports in real time.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="badge-row">'
    '<span class="badge">🔍 Search Agent</span>'
    '<span class="badge">📰 Reader Agent</span>'
    '<span class="badge">✍️ Writer Chain</span>'
    '<span class="badge">🧐 Critic Chain</span>'
    '<span class="badge">⚡ LangGraph</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest advances in solid-state batteries",
        label_visibility="collapsed",
    )
with col2:
    run_clicked = st.button("🚀 Run", type="primary", use_container_width=True,
                             disabled=not topic.strip())

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_clicked:
    state = {}
    start_time = datetime.now()
    progress = st.container()

    with progress:
        with st.status("🔍 Step 1/4 — Searching for sources...", expanded=True) as s1:
            try:
                search_agent = build_search_agent()
                search_result = search_agent.invoke({
                    "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
                })
                state["search_results"] = to_text(search_result["messages"][-1].content)
                st.markdown(f'<div class="agent-card">{state["search_results"][:400]}...</div>',
                            unsafe_allow_html=True)
                s1.update(label="🔍 Step 1/4 — Search complete", state="complete")
            except Exception:
                s1.update(label="🔍 Step 1/4 — Search agent failed", state="error")
                st.exception(traceback.format_exc())
                st.stop()

        with st.status("📰 Step 2/4 — Scraping best source...", expanded=True) as s2:
            try:
                reader_agent = build_reader_agent()
                reader_result = reader_agent.invoke({
                    "messages": [("user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:800]}"
                    )]
                })
                state["scraped_content"] = to_text(reader_result["messages"][-1].content)
                st.markdown(f'<div class="agent-card">{state["scraped_content"][:400]}...</div>',
                            unsafe_allow_html=True)
                s2.update(label="📰 Step 2/4 — Scraping complete", state="complete")
            except Exception:
                s2.update(label="📰 Step 2/4 — Reader agent failed", state="error")
                st.exception(traceback.format_exc())
                st.stop()

        with st.status("✍️ Step 3/4 — Drafting report...", expanded=True) as s3:
            try:
                research_combined = (
                    f"SEARCH RESULTS : \n {state['search_results']} \n\n"
                    f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
                )
                state["report"] = to_text(writer_chain.invoke({
                    "topic": topic, "research": research_combined,
                }))
                st.markdown(f'<div class="agent-card">{state["report"][:400]}...</div>',
                            unsafe_allow_html=True)
                s3.update(label="✍️ Step 3/4 — Draft complete", state="complete")
            except Exception:
                s3.update(label="✍️ Step 3/4 — Writer failed", state="error")
                st.exception(traceback.format_exc())
                st.stop()

        with st.status("🧐 Step 4/4 — Critiquing report...", expanded=True) as s4:
            try:
                state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
                st.markdown(f'<div class="agent-card">{state["feedback"][:400]}...</div>',
                            unsafe_allow_html=True)
                s4.update(label="🧐 Step 4/4 — Review complete", state="complete")
            except Exception:
                s4.update(label="🧐 Step 4/4 — Critic failed", state="error")
                st.exception(traceback.format_exc())
                st.stop()

    elapsed = (datetime.now() - start_time).total_seconds()
    st.session_state.history.append({"topic": topic, "state": state})
    st.session_state.selected = st.session_state.history[-1]
    st.success(f"✅ Pipeline finished in {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
selected = st.session_state.selected
if selected:
    state = selected["state"]
    st.divider()
    st.markdown(f"### 📊 Results — *{selected['topic']}*")

    m1, m2, m3 = st.columns(3)
    m1.metric("Report length", f"{word_count(state.get('report',''))} words")
    m2.metric("Sources scraped", "1")
    m3.metric("Critic feedback", f"{word_count(state.get('feedback',''))} words")

    tab_report, tab_critic, tab_search, tab_scraped = st.tabs(
        ["📄 Report", "🧐 Critic Feedback", "🔍 Search Results", "📰 Scraped Content"]
    )
    with tab_report:
        st.markdown(state.get("report", "_No report generated._"))
        st.download_button("⬇️ Download report (.md)", state.get("report", ""),
                            file_name=f"{selected['topic'][:40]}_report.md")
    with tab_critic:
        st.markdown(state.get("feedback", "_No feedback generated._"))
    with tab_search:
        st.text(state.get("search_results", "_No search results._"))
    with tab_scraped:
        st.text(state.get("scraped_content", "_No scraped content._"))
else:
    if not run_clicked:
        st.info("Enter a topic above and click **Run** to start the research pipeline.")

st.markdown(
    '<div class="footer-note">Nexus Research · Multi-Agent AI Pipeline · '
    'Search → Read → Write → Critique</div>',
    unsafe_allow_html=True,
)
