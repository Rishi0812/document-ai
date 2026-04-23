"""Offline smoke tests for the RAG pipeline (no network calls)."""

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.chunker import (
    _is_table_block,
    _split_table_by_rows,
    _split_with_table_protection,
    chunk_documents,
)
from src.document_loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag_chain import (
    Answer,
    Citation,
    RagChain,
    RetrievedChunk,
    _build_citations,
    _format_context,
    _LLMCitation,
    validate_quote,
)
from src.retriever import build_retriever
from src.vector_store import VectorStoreManager


SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


class _FakeEmbeddings:
    """Deterministic fake embeddings — no API calls."""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        base = [0.0] * 16
        for i, ch in enumerate(text[:16]):
            base[i] = (ord(ch) % 97) / 97.0
        return base


def _fresh_store(tmp_path: Path) -> VectorStoreManager:
    return VectorStoreManager(
        persist_dir=tmp_path / "chroma",
        embeddings=_FakeEmbeddings(),
        collection_name=f"test_{tmp_path.name}",
    )


def test_load_and_chunk_samples():
    paths = list(SAMPLES.glob("*.md"))
    assert paths, "sample markdown files missing"

    docs = load_documents(paths)
    assert len(docs) >= 2
    assert all("source" in d.metadata for d in docs)

    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= len(docs)
    assert all(c.metadata.get("source") for c in chunks)


def test_vector_store_add_list_delete(tmp_path):
    store = _fresh_store(tmp_path)
    store.add(
        [
            Document(page_content="alpha beta", metadata={"source": "a.md", "page": 0}),
            Document(
                page_content="gamma delta", metadata={"source": "b.md", "page": 0}
            ),
        ]
    )
    assert set(store.list_sources()) == {"a.md", "b.md"}
    assert store.count() == 2

    store.delete_source("a.md")
    assert store.list_sources() == ["b.md"]


def test_build_retriever_semantic_mode(tmp_path):
    store = _fresh_store(tmp_path)
    store.add(
        [
            Document(
                page_content="the quick brown fox",
                metadata={"source": "x.md", "page": 0},
            ),
        ]
    )
    retriever = build_retriever(store, mode="semantic", k=1)
    docs = retriever.invoke("fox")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "x.md"


def test_format_context_embeds_citations():
    docs = [
        Document(page_content="hello world", metadata={"source": "a.md", "page": 0}),
        Document(page_content="foo bar", metadata={"source": "b.md", "page": 2}),
    ]
    rendered = _format_context(docs)
    assert "[a.md:1]" in rendered
    assert "[b.md:3]" in rendered


def test_validate_quote_matches_whitespace_and_case_insensitively():
    chunks = [
        RetrievedChunk(
            text="Employees may expense up to $500 per year for additional home-office equipment.",
            source="policy.md",
            page=1,
            score=0.9,
        ),
    ]
    assert (
        validate_quote("employees may expense up to $500 per year", chunks) is not None
    )
    assert (
        validate_quote(
            "Employees may   expense\nup to $500", chunks  # weird whitespace
        )
        is not None
    )
    assert validate_quote("invented text not in chunk", chunks) is None
    assert validate_quote("", chunks) is None


def test_build_citations_drops_hallucinated_quotes():
    chunks = [
        RetrievedChunk(
            text="Passwords must be at least 14 characters long.",
            source="handbook.md",
            page=1,
            score=0.95,
        ),
    ]
    llm_citations = [
        _LLMCitation(
            source="handbook.md",
            page=1,
            quote="Passwords must be at least 14 characters long.",
        ),
        _LLMCitation(
            source="handbook.md",
            page=1,
            quote="Passwords must be rotated hourly.",  # hallucinated
        ),
    ]
    citations = _build_citations(llm_citations, chunks)
    assert len(citations) == 1
    assert citations[0].verified is True
    assert "14 characters" in citations[0].snippet


def test_build_citations_dedupes_same_quote():
    chunks = [
        RetrievedChunk(
            text="Report incidents within 1 hour.",
            source="handbook.md",
            page=1,
            score=0.9,
        ),
    ]
    llm_citations = [
        _LLMCitation(
            source="handbook.md", page=1, quote="Report incidents within 1 hour."
        ),
        _LLMCitation(
            source="handbook.md", page=1, quote="Report incidents within 1 hour."
        ),
    ]
    assert len(_build_citations(llm_citations, chunks)) == 1


def test_is_table_block_detects_markdown_tables():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert _is_table_block(table) is True
    assert _is_table_block("# just a heading") is False
    assert _is_table_block("plain text with | pipes | in it") is False


def test_small_table_kept_atomic_in_chunking():
    md = (
        "# Report\n\n"
        "Some intro paragraph here with a few words to add context.\n\n"
        "| Region | Q1 | Q2 |\n"
        "|--------|----|----|\n"
        "| NA     | 10 | 15 |\n"
        "| EMEA   |  8 | 12 |\n"
        "| APAC   |  5 |  7 |\n\n"
        "Closing remarks about the quarter.\n"
    )
    docs = [Document(page_content=md, metadata={"source": "r.md", "page": 0})]
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=40)

    # The table should appear in exactly one chunk, intact.
    table_chunks = [
        c for c in chunks if "| NA" in c.page_content and "| EMEA" in c.page_content
    ]
    assert len(table_chunks) == 1, f"table was split across chunks: {chunks}"
    assert "| APAC" in table_chunks[0].page_content


def test_large_table_split_by_rows_repeats_header():
    header = "| Region | Value |"
    sep = "|--------|-------|"
    rows = [f"| R{i:03d}   | {i * 10} |" for i in range(40)]
    table = "\n".join([header, sep, *rows])

    slices = _split_table_by_rows(table, max_size=200)
    assert len(slices) > 1  # definitely had to split
    for s in slices:
        # Every slice must start with the header + separator.
        assert s.startswith(header + "\n" + sep), f"missing header in slice: {s[:80]}"

    # Every row must appear somewhere across the slices.
    all_text = "\n".join(slices)
    for row in rows:
        assert row in all_text


def test_split_with_table_protection_mixes_text_and_table():
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
    md = (
        "Intro paragraph one. Intro paragraph two with more words.\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
        "Outro paragraph with some trailing words at the end.\n"
    )
    pieces = _split_with_table_protection(md, splitter, chunk_size=80)
    table_pieces = [p for p in pieces if "| 1 | 2 |" in p and "| 3 | 4 |" in p]
    assert len(table_pieces) == 1  # table stayed together
    assert any("Intro" in p for p in pieces)
    assert any("Outro" in p for p in pieces)


def test_rag_chain_no_docs_returns_grounded_refusal(tmp_path):
    store = _fresh_store(tmp_path)
    fake_llm = MagicMock()
    chain = RagChain(llm=fake_llm, store=store)
    result = chain.answer("unanswerable?", role="default", mode="semantic")
    assert isinstance(result, Answer)
    assert result.confidence == 0.0
    assert "don't have enough information" in result.text
    fake_llm.assert_not_called()
