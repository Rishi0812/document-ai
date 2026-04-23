"""Streamlit UI for the Internal Knowledge Assistant."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.chunker import chunk_documents
from src.config import get_settings
from src.document_loader import is_supported, load_document
from src.rag_chain import Answer, RagChain
from src.vector_store import VectorStoreManager

load_dotenv()

ROLE_LABELS = {
    "default": "General",
    "pm": "Product Manager",
    "sales": "Sales",
    "engineer": "Engineer",
}
MODE_LABELS = {
    "hybrid": "Hybrid",
    "semantic": "Semantic",
    "keyword": "Keyword",
}

st.set_page_config(
    page_title="Internal Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    """Inject custom CSS for a modern look."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _get_api_key() -> str:
    """Return the OpenAI key from Streamlit secrets or env."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")


@st.cache_resource(show_spinner=False)
def _init_store() -> VectorStoreManager:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=_get_api_key()
    )
    return VectorStoreManager(
        persist_dir=settings.persist_dir,
        embeddings=embeddings,
        collection_name=settings.collection_name,
    )


@st.cache_resource(show_spinner=False)
def _init_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model, temperature=0.1, api_key=_get_api_key()
    )


def _ingest_uploads(store: VectorStoreManager, uploaded_files) -> int:
    """Persist uploaded files to disk, load, chunk, and add to the store."""
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    existing = set(store.list_sources())
    added = 0

    for uploaded in uploaded_files:
        if uploaded.name in existing:
            continue

        target = settings.uploads_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())

        if not is_supported(target):
            continue

        with tempfile.TemporaryDirectory():  # isolate load noise
            docs = load_document(target)

        chunks = chunk_documents(
            docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        store.add(chunks)
        added += 1

    return added


def _confidence_color(value: float) -> str:
    if value >= 0.7:
        return "#16A34A"  # green
    if value >= 0.4:
        return "#D97706"  # amber
    return "#DC2626"  # red


def _render_confidence(value: float) -> None:
    pct = int(round(max(0.0, min(1.0, value)) * 100))
    color = _confidence_color(value)
    st.markdown(
        f"""
      <div class="confidence-wrap">
        <div class="confidence-label">Confidence</div>
        <div class="confidence-bar">
          <div class="confidence-fill"
               style="width: {pct}%; background: {color};"></div>
        </div>
        <div class="confidence-value">{pct}%</div>
      </div>
      """,
        unsafe_allow_html=True,
    )


def _render_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    _render_confidence(answer.confidence)

    if answer.citations:
        with st.expander(f"📎 Sources ({len(answer.citations)})", expanded=False):
            for cite in answer.citations:
                st.markdown(
                    f"""
            <div class="citation-card">
              <div class="citation-source">{cite.source} · page {cite.page}</div>
              <div class="citation-snippet">{cite.snippet}</div>
            </div>
            """,
                    unsafe_allow_html=True,
                )

    if answer.chunks:
        with st.expander(f"🔍 Retrieved chunks ({len(answer.chunks)})", expanded=False):
            for idx, chunk in enumerate(answer.chunks, start=1):
                preview = chunk.text.strip().replace("\n", " ")
                if len(preview) > 600:
                    preview = preview[:597] + "..."
                st.markdown(
                    f"""
            <div class="chunk-box">
              <div class="chunk-meta">
                <span>#{idx} · {chunk.source} · page {chunk.page}</span>
                <span class="chunk-score-pill">{chunk.score:.2f}</span>
              </div>
              {preview}
            </div>
            """,
                    unsafe_allow_html=True,
                )


def _sidebar(store: VectorStoreManager) -> tuple[str, str]:
    with st.sidebar:
        st.markdown("### 📚 Knowledge Base")
        st.caption("Upload PDFs, markdown, or text files.")

        uploads = st.file_uploader(
            "Drag & drop files",
            type=["pdf", "md", "markdown", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads:
            with st.spinner("Indexing..."):
                added = _ingest_uploads(store, uploads)
            if added:
                st.success(f"Indexed {added} new file{'s' if added != 1 else ''}.")
            else:
                st.info("All files already indexed.")

        st.divider()

        sources = store.list_sources()
        st.markdown(f"**Indexed documents** ({len(sources)})")
        if not sources:
            st.caption("No documents yet.")
        else:
            for source in sources:
                cols = st.columns([5, 1])
                cols[0].markdown(
                    f'<span class="doc-pill">{source}</span>', unsafe_allow_html=True
                )
                if cols[1].button("✕", key=f"del-{source}", help=f"Remove {source}"):
                    store.delete_source(source)
                    st.cache_resource.clear()
                    st.rerun()

            if st.button("Clear all documents", use_container_width=True):
                store.clear()
                st.cache_resource.clear()  # drop stale Chroma handle
                st.session_state.chat_history = []
                st.rerun()

        st.divider()

        st.markdown("### ⚙️ Response style")
        role_key = st.radio(
            "Role",
            options=list(ROLE_LABELS.keys()),
            format_func=lambda k: ROLE_LABELS[k],
            horizontal=False,
            label_visibility="collapsed",
        )

        st.markdown("### 🔎 Retrieval mode")
        mode_key = st.radio(
            "Mode",
            options=list(MODE_LABELS.keys()),
            format_func=lambda k: MODE_LABELS[k],
            horizontal=True,
            label_visibility="collapsed",
        )

    return role_key, mode_key


def main() -> None:
    _inject_css()

    st.markdown(
        '<div class="hero-title">Internal Knowledge Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">'
        "Ask questions across your documents. Every answer is grounded in your "
        "sources and cited."
        "</div>",
        unsafe_allow_html=True,
    )

    if not _get_api_key():
        st.warning(
            "Set `OPENAI_API_KEY` in `.env` or in Streamlit Cloud secrets to continue."
        )
        st.stop()

    store = _init_store()
    llm = _init_llm()
    chain = RagChain(llm=llm, store=store)

    role, mode = _sidebar(store)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant" and "answer" in turn:
                _render_answer(turn["answer"])
            else:
                st.markdown(turn["content"])

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    if store.count() == 0:
        st.warning("Upload at least one document before asking questions.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.answer(
                question=question, role=role, mode=mode, k=get_settings().top_k
            )
        _render_answer(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "answer": answer, "content": answer.text}
    )


if __name__ == "__main__":
    main()
