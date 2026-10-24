"""
DocuMind — RAG Document Intelligence Assistant
streamlit_app.py

Web UI for the existing RAG pipeline (rag_core.py).

Run with:
    streamlit run streamlit_app.py

Architecture is unchanged:
- same embedding model
- same Chroma DB
- same retriever settings
- same Gemini model and prompt
- API key still comes from .env
"""

import os
import re
import html
import time
import tempfile
from datetime import datetime

import streamlit as st


# ---------------- Page config ----------------

st.set_page_config(
    page_title="DocuMind · Chat with your documents",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------- Load RAG pipeline ----------------

try:
    from rag_core import get_answer, ingest_pdf

    RAG_READY = True
    RAG_ERROR = None

except Exception as e:
    RAG_READY = False
    RAG_ERROR = str(e)


SUGGESTIONS = [
    ("📝", "Summarise the key points"),
    ("🧭", "What are the main topics covered?"),
    ("🔢", "List important dates, numbers or names"),
    ("💡", "What conclusions or recommendations are made?"),
]


# ---------------- Styling (soft indigo, calm but lively) ----------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0e1016;
        --surface: #161a24;
        --surface-2: #1d2230;
        --border: #272d3f;
        --accent: #8b93f8;
        --accent-2: #67d4c4;
        --accent-soft: rgba(139,147,248,0.12);
        --text: #e6e9f2;
        --muted: #8f97ad;
        --ok: #6fcf97;
        --bad: #eb8b8b;
    }

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text); }

    .stApp {
        background:
            radial-gradient(700px 380px at 15% -8%, rgba(139,147,248,0.16), transparent 60%),
            radial-gradient(600px 360px at 95% 8%, rgba(103,212,196,0.09), transparent 60%),
            var(--bg) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden; }
    .block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 6rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: var(--text); }
    .side-brand { font-weight: 700; font-size: 1.2rem; }
    .side-status { display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.78rem; color: var(--ok); margin-top: 0.35rem; }
    .side-status.off { color: var(--bad); }
    .side-status .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    .side-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 0.3rem 0 0.5rem 0; }

    /* ---------- Hero ---------- */
    .hero { text-align: center; margin: 0.5rem 0 1.8rem 0; }
    .logo {
        width: 62px; height: 62px; margin: 0 auto 1rem auto; border-radius: 18px;
        display: flex; align-items: center; justify-content: center; font-size: 1.9rem; font-weight: 800; color: #d5d8ff;
        background: linear-gradient(135deg, rgba(139,147,248,0.30), rgba(103,212,196,0.22));
        border: 1px solid rgba(139,147,248,0.35);
        box-shadow: 0 10px 30px rgba(139,147,248,0.18);
    }
    .hero-title {
        font-size: clamp(2rem, 5vw, 2.9rem); font-weight: 800; letter-spacing: -0.03em; line-height: 1.1;
        background: linear-gradient(120deg, #ffffff 0%, #c4c9ff 55%, #9fe6da 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-sub { color: var(--muted); font-size: 0.98rem; margin-top: 0.55rem; }

    /* ---------- Upload card ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(139,147,248,0.07), rgba(139,147,248,0.02)), var(--surface) !important;
        border: 1px solid var(--border) !important; border-radius: 18px !important;
    }
    .card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.15rem; }
    .card-desc { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.9rem; }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(139,147,248,0.05) !important; border: 1.5px dashed rgba(139,147,248,0.45) !important;
        border-radius: 14px !important; padding: 1.5rem !important; transition: all 0.2s;
    }
    [data-testid="stFileUploaderDropzone"]:hover { background: rgba(139,147,248,0.10) !important; border-color: var(--accent) !important; }
    [data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; }

    .doc-chip {
        display: inline-block; margin: 0.6rem 0.4rem 0 0; padding: 0.22rem 0.7rem; border-radius: 999px;
        font-size: 0.76rem; font-weight: 600; background: var(--accent-soft); color: #b8bdfc; border: 1px solid rgba(139,147,248,0.25);
    }

    /* ---------- Suggestion cards ---------- */
    .section-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 1.6rem 0 0.7rem 0; text-align: center; }
    .sugg .stButton > button {
        background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important;
        border-radius: 14px !important; padding: 0.9rem 1rem !important; font-weight: 500 !important; font-size: 0.88rem !important;
        text-align: left; justify-content: flex-start; min-height: 3.2rem; transition: all 0.18s ease;
    }
    .sugg .stButton > button:hover {
        border-color: var(--accent) !important; background: var(--surface-2) !important;
        transform: translateY(-2px); box-shadow: 0 8px 22px rgba(139,147,248,0.14);
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #8b93f8, #6f78ea) !important; color: #fff !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important; transition: all 0.18s !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 22px rgba(139,147,248,0.3) !important; }
    .stButton > button:focus { box-shadow: none !important; }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] { background: var(--surface) !important; border: 1px solid var(--border); border-radius: 16px; padding: 0.9rem 1.1rem; margin-bottom: 0.4rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, rgba(139,147,248,0.16), rgba(139,147,248,0.07)) !important;
        border-color: rgba(139,147,248,0.3);
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--text); line-height: 1.75; }

    [data-testid="stChatInput"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(139,147,248,0.15); }
    [data-testid="stChatInput"] textarea { color: var(--text) !important; }

    [data-testid="stChatMessageAvatarAssistant"] { background: linear-gradient(135deg, #8b93f8, #67d4c4) !important; }
    [data-testid="stChatMessageAvatarUser"] { background: #3a4260 !important; }

    /* ---------- Sources ---------- */
    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 12px; }
    .source-box {
        background: var(--surface-2); border: 1px solid var(--border); border-left: 3px solid var(--accent-2);
        border-radius: 10px; padding: 0.7rem 0.95rem; margin-bottom: 0.55rem; font-size: 0.83rem; line-height: 1.65; color: #c5cadb;
    }
    .source-head { font-size: 0.72rem; font-weight: 700; color: var(--accent-2); margin-bottom: 0.25rem; }

    hr { border-top: 1px solid var(--border) !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Session state ----------------

defaults = {"messages": [], "ingested_files": [], "pending_query": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- Helpers ----------------

def normalize_answer(answer) -> str:
    """Gemini can return a list of content parts; flatten to plain text."""
    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        answer = "".join(parts)
    return answer if isinstance(answer, str) else str(answer)


def docs_to_sources(docs) -> list:
    sources = []
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        sources.append(
            {
                "source": meta.get("source", "unknown source"),
                "page": meta.get("page"),
                "content": getattr(d, "page_content", str(d)),
            }
        )
    return sources


def render_sources(sources: list):
    with st.expander(f"📚 {len(sources)} source chunk(s) used"):
        for i, s in enumerate(sources, start=1):
            page = f" · p.{int(s['page']) + 1}" if isinstance(s.get("page"), (int, float)) else ""
            body = html.escape(s["content"]).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-head">Chunk {i} · {html.escape(str(s['source']))}{page}</div>
                    {body}
                </div>
                """,
                unsafe_allow_html=True,
            )


def export_chat_md() -> str:
    lines = [f"# DocuMind conversation — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in st.session_state.messages:
        who = "**You**" if m["role"] == "user" else "**DocuMind**"
        lines.append(f"{who}\n\n{m['content']}\n")
        if m.get("sources"):
            lines.append("_Sources:_")
            for s in m["sources"]:
                lines.append(f"- {s['source']}")
            lines.append("")
    return "\n".join(lines)


def stream_text(placeholder, text: str):
    shown = ""
    for token in re.findall(r"\S+\s*", text):
        shown += token
        placeholder.markdown(shown + "▌")
        time.sleep(0.008)
    placeholder.markdown(shown)


def render_upload_card():
    with st.container(border=True):
        st.markdown('<div class="card-title">📎 Add a document</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-desc">Drop a PDF here — it becomes searchable immediately.</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            label_visibility="collapsed",
            disabled=not RAG_READY,
        )

        if uploaded_file is not None:
            if uploaded_file.name in st.session_state.ingested_files:
                st.info(f"'{uploaded_file.name}' was already added this session.", icon="ℹ️")
            else:
                if st.button(f"➕  Add '{uploaded_file.name}'", use_container_width=True):
                    with st.spinner(f"Reading and embedding {uploaded_file.name}..."):
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                tmp_path = tmp.name

                            num_chunks = ingest_pdf(tmp_path, source_name=uploaded_file.name)
                            st.session_state.ingested_files.append(uploaded_file.name)
                            st.toast(f"Added '{uploaded_file.name}' ({num_chunks} chunks indexed)", icon="✅")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to add '{uploaded_file.name}': {e}", icon="🚫")

                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)

        if st.session_state.ingested_files:
            st.markdown(
                "".join(f'<span class="doc-chip">📄 {html.escape(f)}</span>' for f in st.session_state.ingested_files),
                unsafe_allow_html=True,
            )


# ---------------- Sidebar ----------------

with st.sidebar:
    st.markdown('<div class="side-brand">DocuMind</div>', unsafe_allow_html=True)
    if RAG_READY:
        st.markdown('<div class="side-status"><span class="dot"></span>Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-status off"><span class="dot"></span>Offline</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
    show_sources = st.toggle("Show retrieved sources", value=True)

    if st.session_state.messages:
        st.download_button(
            "⬇️  Export conversation",
            data=export_chat_md(),
            file_name="documind_conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- Hero ----------------

st.markdown(
    """
    <div class="hero">
        <div class="logo">D</div>
        <div class="hero-title">DocuMind</div>
        <div class="hero-sub">Chat with your documents — grounded answers with cited sources.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- RAG initialization error ----------------

if not RAG_READY:
    st.error(
        "The RAG pipeline failed to initialize, so chat is disabled.\n\n"
        f"**Details:** {RAG_ERROR}\n\n"
        "Common fixes:\n"
        "- Make sure a `.env` file exists in the project root with `GEMINI_API_KEY=...`\n"
        "- Make sure the `chroma_db` folder exists in the project root and is populated\n"
        "- Make sure all packages from `requirements.txt` are installed",
        icon="🚫",
    )
    st.stop()


# ---------------- Upload (center) ----------------

if st.session_state.messages:
    with st.expander("📎 Add a document", expanded=False):
        render_upload_card()
else:
    render_upload_card()


# ---------------- Chat history ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"])


# ---------------- Suggestions ----------------

if not st.session_state.messages:
    st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sugg">', unsafe_allow_html=True)
    cols = st.columns(2, gap="small")
    for i, (icon, q) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {q}", key=f"sugg_{i}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Chat input ----------------

typed = st.chat_input("Ask a question about your documents...")
query = typed or st.session_state.pending_query
st.session_state.pending_query = None


if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Retrieving context and generating answer..."):
            try:
                answer, docs = get_answer(query)
            except Exception as e:
                answer = f"⚠️ Something went wrong while answering: {e}"
                docs = []

        answer = normalize_answer(answer)
        sources = docs_to_sources(docs)

        stream_text(placeholder, answer)

        if show_sources and sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources or None}
    )
    st.rerun()