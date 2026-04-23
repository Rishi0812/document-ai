"""Persistent Chroma vector store with per-source management."""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """Thin wrapper around Chroma with source-aware add/list/delete."""

    def __init__(
        self,
        persist_dir: Path,
        embeddings: Embeddings,
        collection_name: str = "knowledge_base",
    ):
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._open()

    def _open(self) -> None:
        """(Re)create the underlying Chroma client. Cosine distance → 0..1 scores."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def store(self) -> Chroma:
        return self._store

    def add(self, chunks: list[Document]) -> None:
        """Add chunks. No-op if the list is empty."""
        if not chunks:
            return
        self._store.add_documents(chunks)

    def list_sources(self) -> list[str]:
        """Return unique source filenames currently in the collection."""
        data = self._store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        sources = {m.get("source") for m in metadatas if m and m.get("source")}
        return sorted(sources)

    def count(self) -> int:
        """Return total chunks in the collection."""
        return self._store._collection.count()

    def delete_source(self, source: str) -> None:
        """Delete every chunk whose metadata.source matches `source`."""
        self._store.delete(where={"source": source})

    def clear(self) -> None:
        """Drop the collection and open a fresh, empty one.

        Rebuild-from-scratch is more robust than per-id deletes, which can
        hit SQLite "readonly database" errors if a handle got stale.
        """
        try:
            self._store.delete_collection()
        except Exception:
            # Last resort: nuke the persist dir and reopen.
            try:
                del self._store
            except Exception:
                pass
            shutil.rmtree(self._persist_dir, ignore_errors=True)
        self._open()

    def get_all_chunks(self) -> list[Document]:
        """Return every chunk in the collection as Documents (for BM25)."""
        data = self._store.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas)
        ]
