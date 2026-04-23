"""Load PDF, Markdown, and plain-text files into LangChain Documents.

PDFs are converted to Markdown via `pymupdf4llm`, which detects tables and
renders them as GitHub-flavored markdown tables. This lets the chunker
preserve table structure during splitting.
"""

from pathlib import Path

import pymupdf4llm
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def is_supported(path: Path) -> bool:
    """Return True if the file extension has a registered loader."""
    suffix = path.suffix.lower()
    return suffix == ".pdf" or suffix in _TEXT_SUFFIXES


def _load_pdf(path: Path) -> list[Document]:
    """Convert a PDF to Markdown, one Document per page, tables preserved."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    docs: list[Document] = []
    for idx, page in enumerate(pages):
        text = page.get("text", "") if isinstance(page, dict) else str(page)
        if not text or not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": idx},
            )
        )
    return docs


def _load_text(path: Path) -> list[Document]:
    """Load a markdown or text file as a single Document."""
    docs = TextLoader(str(path)).load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata.setdefault("page", 0)
    return docs


def load_document(path: Path) -> list[Document]:
    """Load a single file; one Document per PDF page, or one per md/txt file."""
    if not is_supported(path):
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    return _load_text(path)


def load_documents(paths: list[Path]) -> list[Document]:
    """Load every supported file in `paths`."""
    results: list[Document] = []
    for path in paths:
        if is_supported(path):
            results.extend(load_document(path))
    return results
