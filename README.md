# Internal Knowledge Assistant (RAG)

A retrieval-augmented-generation (RAG) app that lets teams upload internal
documents — PDFs (including ones with real tables), Markdown, or plain text —
and ask questions about them conversationally. Every answer is grounded in
the uploaded sources, backed by **verbatim quotes that the LLM actually used**,
and validated against the indexed content.

---

## Features

- Multi-file upload (**PDF, Markdown, TXT**) with persistent local indexing
- **Structure-aware ingestion** — PDFs convert to Markdown with detected
  tables preserved as `|---|` blocks
- **Intelligent chunking** — splits at markdown headers first, then enforces
  size limits while keeping tables atomic (or split by row-groups with the
  header row repeated so every slice is readable standalone)
- **Three retrieval modes** — semantic (cosine), keyword (BM25), hybrid
  ensemble
- **Role-based responses** — General, PM, Sales, Engineer (system-prompt switch)
- **Structured LLM output with quote validation** — the model returns exact
  quotes alongside its answer; each quote is verified by substring match
  against retrieved chunks before being shown as a citation
- **Retrieved-chunk preview** with question-relative relevance scores
- **Confidence bar** tied to the top chunk's relevance to the question
- Modern Streamlit UI (custom theme + CSS, card-style citations)

## Architecture

```
PDF / MD / TXT
      │
      ▼
┌──────────────────┐   pymupdf4llm (PDFs → Markdown, tables → |---|)
│  DocumentLoader  │   TextLoader   (MD / TXT pass-through)
└──────────────────┘
      │  Documents (page_content: markdown, metadata: {source, page})
      ▼
┌──────────────────┐   MarkdownHeaderTextSplitter  (structural)
│     Chunker      │   → table-preserving RecursiveCharacterTextSplitter
└──────────────────┘   (atomic small tables, row-group split for big ones)
      │  chunks (metadata: {source, page, section, h1/h2/h3})
      ▼
┌──────────────────┐   OpenAI text-embedding-3-small → cosine-distance Chroma
│   Vector store   │   (per-source add / list / delete / clear)
└──────────────────┘
      │
      ▼
User question
      │
      ▼
┌──────────────────┐   semantic | BM25 | hybrid (EnsembleRetriever, 0.4/0.6)
│    Retriever     │
└──────────────────┘
      │
      ▼
┌──────────────────┐   ChatPromptTemplate(role) → ChatOpenAI gpt-4o-mini
│   Structured     │     .with_structured_output(_LLMAnswer)
│    RAG chain     │   → answer text + citation quotes
│                  │   → validate each quote via normalized substring match
└──────────────────┘   → drop hallucinated quotes; return Answer(...)
      │
      ▼
      Answer(text, citations[], retrieved_chunks[], confidence)
```

Every module in `src/` is small, pure, and independently testable:

| File | Responsibility |
|---|---|
| [src/config.py](src/config.py) | Typed settings from env (`pydantic-settings`) |
| [src/document_loader.py](src/document_loader.py) | PDFs via **pymupdf4llm** (→ markdown with tables), MD/TXT via TextLoader |
| [src/chunker.py](src/chunker.py) | `MarkdownHeaderTextSplitter` → table-preserving size splitter |
| [src/vector_store.py](src/vector_store.py) | Chroma wrapper (cosine distance, robust add/list/delete/clear) |
| [src/retriever.py](src/retriever.py) | Semantic / BM25 / Hybrid (`EnsembleRetriever`) |
| [src/rag_chain.py](src/rag_chain.py) | Retrieval → structured LLM output → verbatim-quote validation → `Answer` |
| [src/prompts.py](src/prompts.py) | Role-specific system prompts + citation-enforcing QA template |
| [app.py](app.py) | Streamlit UI only — no business logic |

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | OpenAI `gpt-4o-mini` | Cheap, fast, strong RAG quality |
| Embeddings | OpenAI `text-embedding-3-small` | Inexpensive, high quality |
| PDF extraction | **pymupdf4llm** | Local, fast, renders tables as Markdown |
| Framework | LangChain 1.2 | Loaders, splitters, retrievers, structured output |
| Vector store | **Chroma (cosine)** | Persistent local; scores map cleanly into `[0, 1]` |
| UI | Streamlit 1.56 | Shortest path to a stunning hosted demo |
| Config | `pydantic-settings` + `.env` | Typed, explicit, 12-factor |

Full pinned list in [requirements.txt](requirements.txt).

---

## Document support — how each file type is handled

The goal is that **every document, regardless of format, becomes a stream of
clean Markdown** before it touches the embedding model. That gives us one
splitting strategy for all file types.

| Format | Loader | Notes |
|---|---|---|
| **PDF** | `pymupdf4llm.to_markdown(path, page_chunks=True)` | One `Document` per page. Tables detected via PyMuPDF's `find_tables()` and rendered as GFM `\|---\|` tables. Headings preserved. No OCR; scanned PDFs need to be OCR'd upstream. |
| **Markdown** (`.md`, `.markdown`) | `TextLoader` (read as text) | Markdown syntax is already LLM-friendly; no parsing step needed. Header splitter downstream handles the structure. |
| **Plain text** (`.txt`) | `TextLoader` | Loaded as-is; falls back to recursive character splitting since there are no headers to split on. |

### What the intelligent chunker does

1. **Structural split** — `MarkdownHeaderTextSplitter` breaks documents at
   `#`, `##`, and `###` boundaries. A section (e.g. *"2. Leave & Time Off"*)
   becomes one chunk when it fits, keeping related content together instead
   of arbitrarily cutting mid-paragraph. The header trail (h1 > h2 > h3) is
   saved to `metadata["section"]` for downstream attribution.
2. **Size enforcement with table protection** — sections longer than
   `chunk_size` (default 500 chars) are re-split by `RecursiveCharacterTextSplitter`.
   Before that size-based pass runs, we detect markdown tables with a regex and:
   - **Small table** (≤ `chunk_size × 1.5`): passed through as a single atomic
     chunk. No table is ever cut in half.
   - **Large table**: sliced by row groups; the header row + separator row are
     **repeated at the top of every slice**, so the LLM always sees column
     names and each chunk is independently interpretable.
3. **Metadata propagation** — every resulting chunk carries `source`, `page`,
   `section`, and the raw `h1/h2/h3` levels so citations can be specific.

### Why this matters

Without structure-aware chunking, a question like *"How many employees are
based in Bangalore?"* would fail: the naive splitter cuts a 20-row employee
table mid-stream, retrieval returns half the rows, and the model hedges.
With the header-repeat strategy each slice starts with `| Name | Department |
Role | Location | ... |`, so every slice is a self-contained mini-table and
retrieval can find both Bangalore rows even when they're in different chunks.

## Grounding guarantees

Answers are grounded in two layers:

1. **Structured LLM output.** The chain uses
   `llm.with_structured_output(_LLMAnswer)` to get back a Pydantic object:
   `{text, citations: [{source, page, quote}]}`. The prompt instructs the
   model to copy each `quote` **verbatim** from the context block it came
   from, with one to three sentences per quote.
2. **Quote validation.** [`validate_quote()`](src/rag_chain.py) normalizes
   whitespace and case, then checks that each model-supplied quote is a
   substring of one of the retrieved chunks. If a quote doesn't match
   (paraphrase, invention, wrong filename), the citation is **dropped**.
   The UI only shows citations that survived this check, so the user can
   trust the "Sources" card reflects actual text in the document.

The `text` field still includes inline `[filename:page]` tags for
readability; the validated `citations` list powers the Sources card.

## Retrieval & scoring

- **Chroma with cosine distance** (`collection_metadata={"hnsw:space": "cosine"}`)
  so relevance scores land cleanly in `[0, 1]`.
- **Question-relative scoring.** Earlier versions accidentally scored each
  retrieved chunk against *itself* (always ≈1.0). The current code runs a
  separate `similarity_search_with_relevance_scores(question, k=2k)` and
  maps those scores onto the retrieval result. Ordering is by descending
  relevance.
- **Low-relevance filter.** Chunks scoring below `max(0.3, top × 0.6)` are
  dropped before rendering — prevents the Retrieved-chunks panel from being
  cluttered with off-topic matches (especially when the index is small and
  `top_k` pulls in everything).
- **Confidence = top chunk's score.** Single high-quality match → green
  bar. A tail of weak matches no longer drags the bar down.
- **Hybrid = BM25 ⊕ semantic** via `EnsembleRetriever(weights=[0.4, 0.6])`.
  Semantic catches paraphrases; BM25 catches exact identifiers (error codes,
  policy numbers). Users can toggle between the three modes in the sidebar.

## Stack

| Layer | Choice | Version |
|---|---|---|
| LLM | OpenAI `gpt-4o-mini` | via `langchain-openai` 1.2 |
| Embeddings | OpenAI `text-embedding-3-small` | |
| PDF → Markdown | `pymupdf4llm` | 1.27.2 |
| Framework | `langchain` + `langchain-classic` | 1.2 / 1.0 |
| Vector store | `chromadb` + `langchain-chroma` | 1.5 / 1.1 |
| UI | `streamlit` + `streamlit-extras` | 1.56 |
| Config | `pydantic-settings` | 2.14 |

Pins in [requirements.txt](requirements.txt).

---

## Local setup

```bash
# 1. Use the project's conda env
conda create -n document-ai python=3.11 -y  # skip if already present
conda activate document-ai

# 2. Install pinned deps
pip install -r requirements.txt

# 3. Configure your OpenAI key
cp .env.example .env
# edit .env — set OPENAI_API_KEY=sk-...

# 4. Run
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

**Quick test:** upload the three files in `data/samples/`:

- [remote_work_policy.md](data/samples/remote_work_policy.md)
- [security_handbook.md](data/samples/security_handbook.md)
- [employee_handbook.pdf](data/samples/employee_handbook.pdf) — multi-page
  handbook with real structured tables

Then try these questions:

| Question | Expected |
|---|---|
| *"Which department does Priya Nair work in?"* | Product / Principal PM — cites the exact table row |
| *"How many employees are based in Bangalore?"* | **2** — cites both rows |
| *"What's the sabbatical policy?"* | 4 weeks every 5 years, manager approval (leave table row) |
| *"How many parental leave weeks does a primary caregiver get?"* | 16 weeks |
| *"Can I get an extra amount for home equipment?"* | $500/year |
| *"How long do passwords need to be?"* | 14 characters |
| *"Who won the 2022 FIFA World Cup?"* | *"I don't have enough information…"* |

## Tests

```bash
pytest tests/ -v
```

The offline suite (12 tests, no network) covers:

- Document loading (PDF + MD + TXT)
- Chunker — header splits, atomic small tables, row-group splitting for
  large tables with header repetition, mixed text/table interleaving
- Vector store add/list/delete
- Retriever construction (semantic mode)
- Context formatting with `[source:page]` tags
- Quote validation — tolerates whitespace/case, rejects hallucinations,
  dedupes repeated citations
- Grounded refusal path when no docs are indexed

A separate live-API script ([tests/e2e_check.py](tests/e2e_check.py))
exercises the full pipeline against real OpenAI — run with
`python tests/e2e_check.py` when you want to verify end-to-end answer
quality including table reasoning.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → **New app** → pick this repo and `app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy. Paste the resulting `*.streamlit.app` URL at the top of this README.

## Design trade-offs

Each decision below is one we'd be asked about in a review. The reasoning
is captured so a future maintainer can revisit it intelligently.

- **pymupdf4llm over docling / marker-pdf.** We need PDF → Markdown with
  proper tables. `docling` has the best table structure recovery (IBM's
  TableFormer model) and `marker-pdf` is a strong middle ground, but both
  pull in hundreds of MB of ML weights. `pymupdf4llm` is small, fast,
  license-compatible, and good enough for digital PDFs. For scanned
  documents or complex multi-column layouts we'd swap in `docling` — the
  `DocumentLoader` interface is the seam.
- **Markdown as the lingua franca.** Every format converges to Markdown
  before chunking, so we get exactly one splitter to maintain. Tables
  round-trip cleanly; headings map to the structural splitter; LLMs read
  Markdown natively.
- **Header-first then size split.** Splitting at section boundaries before
  enforcing size gives dramatically better retrieval on structured
  documents (policies, handbooks, RFCs) than pure recursive splitting.
- **Tables are atomic.** Cutting a table in half makes retrieval useless —
  you lose the column headers for half the rows. Keeping tables whole (or
  repeating the header across row groups) was the single biggest win for
  fact-lookup questions on the employee-directory PDF.
- **LangChain structured output + substring validation.** Asking the LLM to
  return quotes and then verifying them removes a whole class of
  "citation hallucination" bugs. Earlier iterations relied on a heuristic
  "best sentence in chunk" picker; the structured-output version is
  simpler, more faithful, and rejects paraphrases outright.
- **Cosine distance in Chroma.** Default L2 distance produced negative
  "relevance" scores that broke our `[0, 1]` UX. Switching to cosine costs
  nothing and gives meaningful confidence bars.
- **Question-relative scoring.** Scoring retrieved chunks against the
  *query*, not against themselves, fixed a bug where every chunk scored
  1.00. Now confidence actually varies with how well the corpus answers
  the question.
- **Hybrid retrieval on by default.** Pure semantic misses exact-match
  tokens (error codes, employee names); pure BM25 misses paraphrases.
  Ensemble (0.4 / 0.6) is a robust default; the UI lets users toggle for
  comparison.
- **Grounded refusal prompt.** The QA template is tight enough to prevent
  invention but "charitable" enough to match *"extra for home equipment"*
  to *"additional home-office equipment"*. Achieved this by including
  synonym examples directly in the prompt.
- **`Clear all documents` rebuilds the collection.** Earlier versions did
  per-ID deletes, which hit SQLite *"readonly database"* errors when a
  cached Chroma handle went stale across Streamlit reruns. `clear()` now
  drops the entire collection and reopens it; `st.cache_resource.clear()`
  is called in tandem so the UI never keeps a dead handle.
- **Streamlit over FastAPI + React.** One-file deploy, free hosting, fast
  iteration. We'd revisit only if this became a real multi-tenant product.
- **Only `gpt-4o-mini` is used.** Cheapest OpenAI chat model that still
  handles table reasoning well. Centralized in [`src/config.py`](src/config.py)
  as `chat_model`; swap one line to change.

## Known limitations

- **Not a real conversation — no memory.** Each question is answered in
  isolation. The chat history you see in the UI is render-only; it is not
  fed back into the retriever or the LLM. Follow-up questions that depend
  on a prior turn (*"and what about EMEA?"* after asking about APAC) won't
  resolve the pronoun. Adding memory is the obvious next step — either a
  LangChain `RunnableWithMessageHistory` wrapper or a question-rewriting
  pre-step that uses the last N turns to reformulate an ambiguous
  follow-up into a standalone query before retrieval.
- **Scanned PDFs are not OCR'd.** `pymupdf4llm` relies on extractable
  text. For scanned scans, run `ocrmypdf` or swap to `docling` (has OCR).
- **Table extraction is Markdown-only.** `pymupdf4llm` emits tables as
  GFM `|---|` markdown, which loses richer structure (merged cells,
  multi-row headers, nested tables). LLMs often reason better over
  explicit HTML (`<thead>`, `<th>`, `rowspan`, `colspan`). With more
  budget / infra we would swap the loader for a stronger extractor —
  the `DocumentLoader` interface is the seam. Candidates and why:
    - **Docling** (IBM, open-source) — native HTML export + dedicated
      TableFormer model for spanning cells and multi-row headers.
      Right choice if we want to stay local and open-source.
    - **Mistral OCR** — cheap, fast hosted OCR with strong layout
      understanding; good for scanned / image-heavy PDFs we can't
      handle today.
    - **Azure Document Intelligence** — enterprise-grade layout + table
      extraction, pre-built models for invoices / receipts / IDs,
      strong SLAs; right call for regulated customers already on Azure.
    - **AWS Textract** — similar to Azure, very reliable table detection
      (`FORMS` + `TABLES` features), scales well; fits customers who
      are AWS-native.
  All four preserve structure we currently drop (spanning cells, typed
  headers, form key-value pairs), which would tighten citation snippets
  and make large-table retrieval more precise.
- **Single-process Chroma.** One collection shared across Streamlit
  sessions. Fine for a demo; production needs per-user isolation.
- **Confidence is heuristic**, not a calibrated probability.
- **No cross-encoder re-ranking** (e.g. Cohere rerank, BGE reranker). The
  natural next step above hybrid retrieval.
- **No streaming** — answers render after the full LCEL invocation.
- **Streamlit Cloud cold starts** — first request after idle can be 10–30s.
## Project layout

```
document-ai/
├── app.py                       # Streamlit UI
├── src/
│   ├── config.py
│   ├── document_loader.py       # pymupdf4llm + TextLoader
│   ├── chunker.py               # MarkdownHeader + table-aware
│   ├── vector_store.py          # Chroma (cosine)
│   ├── retriever.py             # semantic / keyword / hybrid
│   ├── rag_chain.py             # structured output + quote validation
│   └── prompts.py               # role prompts + QA template
├── assets/styles.css
├── .streamlit/config.toml
├── data/samples/                # demo documents to try
├── tests/
│   ├── test_rag.py              # 12 offline tests
│   └── e2e_check.py             # live OpenAI script
├── requirements.txt
└── .env.example
```
