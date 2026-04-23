"""RAG chain producing grounded answers with validated citation quotes."""

import re
import warnings
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Chroma occasionally returns relevance scores slightly outside [0,1] due to
# floating-point drift with cosine distance. We clamp on our side; the warning
# is noise for the end user.
warnings.filterwarnings(
    "ignore",
    message="Relevance scores must be between 0 and 1",
)

from src.prompts import QA_TEMPLATE, get_system_prompt
from src.retriever import RetrievalMode, build_retriever
from src.vector_store import VectorStoreManager

_WS_RE = re.compile(r"\s+")


# --- LLM-facing schema (structured output) ---------------------------------


class _LLMCitation(BaseModel):
    """Schema the LLM fills in for one supporting quote."""

    source: str = Field(description="Filename exactly as shown in the context tag.")
    page: int = Field(description="Page number exactly as shown in the context tag.")
    quote: str = Field(
        description=(
            "Verbatim substring copied character-for-character from the context "
            "block. One to three sentences long. Do NOT paraphrase."
        )
    )


class _LLMAnswer(BaseModel):
    """Structured response from the LLM."""

    text: str = Field(description="Answer with inline [filename:page] citations.")
    citations: list[_LLMCitation] = Field(
        default_factory=list,
        description="Supporting spans, one per claim.",
    )


# --- Public return types ---------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single retrieved chunk and its similarity score (0..1)."""

    text: str
    source: str
    page: int
    score: float


@dataclass
class Citation:
    """One citation backing the answer, with a verified verbatim quote."""

    source: str
    page: int
    snippet: str
    verified: bool = True


@dataclass
class Answer:
    """End-to-end RAG result."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0


# --- Helpers ---------------------------------------------------------------


def _page_of(doc: Document) -> int:
    """Return 1-indexed page number from document metadata."""
    raw = doc.metadata.get("page", 0)
    try:
        return int(raw) + 1
    except (TypeError, ValueError):
        return 1


def _format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a numbered context block for the prompt."""
    lines = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = _page_of(doc)
        lines.append(f"[{source}:{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(lines)


def _scored_chunks(
    store: VectorStoreManager,
    docs: list[Document],
    question: str,
) -> list[RetrievedChunk]:
    """Attach question-relative relevance scores, sorted by descending score.

    Queries Chroma with the question to get normalized relevance scores,
    then maps them back onto the retrieved docs (which may come from BM25,
    semantic, or a hybrid ensemble). Retrieved docs with no semantic score
    fall back to 0.0.
    """
    if not docs:
        return []

    score_by_text: dict[str, float] = {}
    try:
        pairs = store.store.similarity_search_with_relevance_scores(
            question, k=max(len(docs) * 2, 8)
        )
        for doc, score in pairs:
            score_by_text[doc.page_content] = float(score)
    except Exception:
        pass

    chunks = [
        RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=_page_of(doc),
            score=max(0.0, min(1.0, score_by_text.get(doc.page_content, 0.0))),
        )
        for doc in docs
    ]
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def _filter_low_relevance(
    chunks: list[RetrievedChunk],
    min_score: float = 0.3,
    min_keep: int = 1,
) -> list[RetrievedChunk]:
    """Drop chunks far below the top score, keeping at least `min_keep`.

    A chunk is dropped if its score is < max(min_score, top_score * 0.6).
    This trims obviously-unrelated retrievals while keeping meaningful
    secondary matches.
    """
    if len(chunks) <= min_keep:
        return chunks
    top = chunks[0].score
    threshold = max(min_score, top * 0.6)
    kept = [c for c in chunks if c.score >= threshold]
    return kept if len(kept) >= min_keep else chunks[:min_keep]


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for tolerant substring matching."""
    return _WS_RE.sub(" ", text).strip().lower()


def validate_quote(quote: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Return the chunk that contains `quote` verbatim, else None.

    Matching is whitespace- and case-insensitive but otherwise exact — no
    fuzzy similarity. If the LLM paraphrased the quote or invented it, this
    returns None and the citation is rejected.
    """
    needle = _normalize(quote)
    if not needle:
        return None
    for chunk in chunks:
        if needle in _normalize(chunk.text):
            return chunk
    return None


def _build_citations(
    llm_citations: list[_LLMCitation],
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Validate each LLM-provided quote against retrieved chunks."""
    seen: set[tuple[str, int, str]] = set()
    citations: list[Citation] = []
    for entry in llm_citations:
        match = validate_quote(entry.quote, chunks)
        if match is None:
            continue  # drop hallucinated or paraphrased quotes
        key = (match.source, match.page, _normalize(entry.quote))
        if key in seen:
            continue
        seen.add(key)
        snippet = entry.quote.strip().replace("\n", " ")
        citations.append(
            Citation(
                source=match.source,
                page=match.page,
                snippet=snippet,
                verified=True,
            )
        )
    return citations


# --- Chain -----------------------------------------------------------------


class RagChain:
    """Assemble retriever + prompt + structured LLM into a single call."""

    def __init__(self, llm: ChatOpenAI, store: VectorStoreManager):
        self._llm = llm
        self._store = store

    def answer(
        self,
        question: str,
        role: str = "default",
        mode: RetrievalMode = "hybrid",
        k: int = 4,
    ) -> Answer:
        """Retrieve context, invoke structured LLM, return grounded Answer."""
        retriever = build_retriever(self._store, mode=mode, k=k)
        docs = retriever.invoke(question)

        if not docs:
            return Answer(
                text=(
                    "I don't have enough information in the provided documents to "
                    "answer that."
                ),
                citations=[],
                chunks=[],
                confidence=0.0,
            )

        chunks = _filter_low_relevance(_scored_chunks(self._store, docs, question))
        # Confidence = best chunk's relevance to the question.
        confidence = chunks[0].score if chunks else 0.0

        prompt = ChatPromptTemplate.from_messages(
            [("system", get_system_prompt(role)), ("user", QA_TEMPLATE)]
        )
        structured_llm = self._llm.with_structured_output(_LLMAnswer)
        chain = prompt | structured_llm
        result: _LLMAnswer = chain.invoke(
            {"context": _format_context(docs), "question": question}
        )

        return Answer(
            text=result.text.strip(),
            citations=_build_citations(result.citations, chunks),
            chunks=chunks,
            confidence=round(confidence, 3),
        )
