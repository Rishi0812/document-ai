"""End-to-end verification against the real OpenAI API.

Not run by default (not named `test_*`). Invoke directly:
    python tests/e2e_check.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.chunker import chunk_documents
from src.config import get_settings
from src.document_loader import load_documents
from src.rag_chain import RagChain
from src.vector_store import VectorStoreManager

ROOT = Path(__file__).resolve().parents[1]


def _header(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def main() -> None:
    load_dotenv()
    settings = get_settings()

    scratch = ROOT / "storage" / "chroma_e2e"
    if scratch.exists():
        shutil.rmtree(scratch)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    llm = ChatOpenAI(model=settings.chat_model, temperature=0.1)
    store = VectorStoreManager(
        persist_dir=scratch, embeddings=embeddings, collection_name="e2e"
    )

    _header("1. Ingest")
    sample_dir = ROOT / "data" / "samples"
    sample_paths = [
        *sample_dir.glob("*.md"),
        *sample_dir.glob("*.pdf"),
    ]
    print(f"Loading {len(sample_paths)} files: {[p.name for p in sample_paths]}")
    docs = load_documents(sample_paths)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store.add(chunks)
    print(f"Indexed {store.count()} chunks across {store.list_sources()}")

    chain = RagChain(llm=llm, store=store)
    questions = [
        (
            "Which department does Priya Nair work in and what's her role?",
            "default",
            "hybrid",
        ),
        (
            "How many employees in the directory are based in Bangalore?",
            "default",
            "hybrid",
        ),
        ("What's the sabbatical policy?", "default", "hybrid"),
        (
            "How many parental leave weeks does a primary caregiver get?",
            "default",
            "semantic",
        ),
        ("Can I get extra amount for home equipment?", "default", "hybrid"),
        ("How long do passwords need to be?", "engineer", "semantic"),
        ("Who won the 2022 FIFA World Cup?", "default", "hybrid"),
    ]

    for i, (q, role, mode) in enumerate(questions, start=1):
        _header(f"{i}. [{mode}/{role}] {q}")
        answer = chain.answer(q, role=role, mode=mode)
        print(f"\nAnswer:\n{answer.text}")
        print(f"\nConfidence: {answer.confidence}")
        print(f"Citations: {len(answer.citations)}")
        for c in answer.citations[:3]:
            snippet = c.snippet[:120].replace("\n", " ")
            print(f"  - {c.source}:{c.page} — {snippet}")

    shutil.rmtree(scratch)
    print("\n✅ E2E run complete.")


if __name__ == "__main__":
    main()
