"""Structure-aware chunking: markdown headers + table-preserving size split.

Strategy:
1. First pass — split each document at markdown headers (#, ##, ###). This
   keeps semantically related content together (a whole section in one piece)
   instead of slicing mid-paragraph.
2. Second pass — enforce chunk_size, but treat markdown tables as atomic
   units. If a table fits inside `chunk_size * 1.5`, it passes through whole;
   larger tables are split by row groups with the header row repeated at
   the top of each slice so every chunk is still readable in isolation.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_TABLE_BLOCK_RE = re.compile(
    r"(^\s*\|[^\n]+\|\s*\n\s*\|[\s:|-]+\|\s*\n(?:\s*\|[^\n]*\|\s*\n?)+)",
    re.MULTILINE,
)


def _is_table_block(text: str) -> bool:
    """True if `text` looks like a markdown table (row + separator + row)."""
    return bool(re.match(r"^\s*\|.+\|\s*\n\s*\|[\s:|-]+\|", text))


def _split_table_by_rows(table: str, max_size: int) -> list[str]:
    """Split a large markdown table into chunks that repeat the header row."""
    lines = [ln for ln in table.strip().split("\n") if ln.strip()]
    if len(lines) < 3:
        return [table.strip()]

    header, separator, *rows = lines
    head = f"{header}\n{separator}"

    slices: list[str] = []
    current: list[str] = []
    current_size = len(head)
    for row in rows:
        projected = current_size + len(row) + 1
        if current and projected > max_size:
            slices.append(head + "\n" + "\n".join(current))
            current, current_size = [row], len(head) + len(row) + 1
        else:
            current.append(row)
            current_size = projected
    if current:
        slices.append(head + "\n" + "\n".join(current))
    return slices


def _split_with_table_protection(
    text: str,
    text_splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[str]:
    """Split `text` while keeping markdown tables intact (or row-group sliced)."""
    results: list[str] = []
    cursor = 0
    for match in _TABLE_BLOCK_RE.finditer(text):
        before = text[cursor : match.start()].strip()
        if before:
            results.extend(text_splitter.split_text(before))

        table = match.group(1).strip()
        if len(table) <= int(chunk_size * 1.5):
            results.append(table)
        else:
            results.extend(_split_table_by_rows(table, chunk_size))
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        results.extend(text_splitter.split_text(tail))
    return results


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    """Produce structure-aware chunks with tables preserved.

    Each input Document is first split at markdown headers (so sections stay
    together), then re-split to fit `chunk_size` with tables kept atomic.
    Metadata (source, page) is propagated to every output chunk; header
    trail (h1 > h2 > h3) is added as `metadata['section']` when available.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out: list[Document] = []
    for doc in docs:
        try:
            sections = header_splitter.split_text(doc.page_content)
        except Exception:
            sections = [doc]

        if not sections:
            sections = [doc]

        for section in sections:
            base_meta = {**doc.metadata, **(section.metadata or {})}
            trail = " > ".join(
                v for k, v in base_meta.items() if k in ("h1", "h2", "h3") and v
            )
            if trail:
                base_meta["section"] = trail

            pieces = _split_with_table_protection(
                section.page_content, size_splitter, chunk_size
            )
            for piece in pieces:
                if not piece.strip():
                    continue
                out.append(Document(page_content=piece, metadata=dict(base_meta)))
    return out
