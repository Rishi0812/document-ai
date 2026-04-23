"""Retrieval strategies: semantic, keyword (BM25), and hybrid."""

from typing import Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from src.vector_store import VectorStoreManager

RetrievalMode = Literal["semantic", "keyword", "hybrid"]


def build_retriever(
    store: VectorStoreManager,
    mode: RetrievalMode = "hybrid",
    k: int = 4,
) -> BaseRetriever:
    """Build a retriever for the given mode.

    - `semantic`: Chroma vector similarity.
    - `keyword`:  BM25 over all indexed chunks.
    - `hybrid`:   EnsembleRetriever blending BM25 (0.4) + semantic (0.6).
    """
    if mode == "semantic":
        return store.store.as_retriever(search_kwargs={"k": k})

    chunks = store.get_all_chunks()
    if not chunks:
        # Nothing indexed yet — fall back to empty semantic retriever.
        return store.store.as_retriever(search_kwargs={"k": k})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    if mode == "keyword":
        return bm25

    semantic = store.store.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.4, 0.6])
