# Implementation Plan — Multi-modal RAG Document Analyst

> A complete record of the design decisions, build phases, technical choices, and evolution of this project from initial idea to finished product.

---

## 1. Problem Statement

Modern knowledge work involves reading, extracting, and reasoning over large volumes of PDF documents — reports, research papers, manuals, slide decks. Standard search and keyword matching fails when:

- Key information is inside **charts, diagrams, or scanned images**
- Questions require **reasoning across multiple pages** or sections
- Users want to interrogate documents in **natural language**, not search queries
- Privacy constraints prevent sending documents to cloud AI services

**Goal:** Build a fully local, multi-modal document QA system that understands both text and visual content in PDFs.

---

## 2. Design Goals & Constraints

| Goal | Detail |
|---|---|
| **100% local / offline** | No API keys, no data leaves the machine |
| **Multi-modal** | Index text AND visuals (charts, tables, images) |
| **Multi-document** | Support a library of many PDFs simultaneously |
| **Per-document scoping** | Query a single document or all at once |
| **Persistent storage** | Knowledge base survives app restarts |
| **GPU memory safe** | Work on consumer GPUs (4–8 GB VRAM) |
| **Beautiful UI** | Premium dark-themed Streamlit interface |
| **No Poppler dependency** | Pure PyMuPDF for all PDF operations |

---

## 3. Technology Stack Decision

### Language Model Stack

| Component | Choice | Rationale |
|---|---|---|
| LLM runner | **Ollama** | Single binary, local GPU support, model management |
| Text LLM | **Llama 3 (8B)** | Strong reasoning, instruction following, 8k context |
| Vision LLM | **Moondream** | Lightweight VLM that runs on 4 GB VRAM |
| Embeddings | **all-MiniLM-L6-v2** | Fast, CPU-based, 384-dim, excellent semantic quality |

### Data Layer

| Component | Choice | Rationale |
|---|---|---|
| Vector store | **ChromaDB (persistent)** | Embedded, no server needed, supports metadata filters |
| PDF parsing | **PyMuPDF (fitz)** | Fast, no external dependencies (replaces Poppler) |
| Image format | **PNG (150 DPI)** | Good quality for VLM, reasonable file size |

### Application Layer

| Component | Choice | Rationale |
|---|---|---|
| UI framework | **Streamlit** | Rapid prototyping, Python-native, easy to deploy |
| Session management | `st.session_state` | Persistent chat history within a session |
| Progress feedback | `st.progress` + `st.empty` | Real-time ingestion status |

---

## 4. Architecture Design

### Module Decomposition

```
app.py          ← Streamlit UI, orchestration, session state
  │
  ├── ingest.py       ← PDF ingestion pipeline (text + images)
  │     └── resources.py   ← Shared singletons (loaded once)
  │
  └── query_engine.py ← Retrieval + LLM answering
        └── resources.py   ← Same shared singletons
```

**Key design choice: shared `resources.py` singleton pattern**  
Loading `SentenceTransformer` and `ChromaDB` client on every function call would be ~3 seconds of latency. By using module-level globals with lazy initialisation, models load exactly once per process lifetime.

### Data Schema (ChromaDB)

Each chunk stored in ChromaDB has:

```json
{
  "id": "<uuid>",
  "document": "<text content or VLM description>",
  "embedding": [384-dimensional float vector],
  "metadata": {
    "page": 3,
    "type": "text" | "image",
    "source": "filename.pdf",
    "image_path": "/path/to/page_3.png"   // only for image chunks
  }
}
```

The `source` field enables **per-document scoped search** via ChromaDB's `where=` filter.

---

## 5. Implementation Phases

### Phase 1 — Core Ingestion Pipeline

**Goal:** Extract and store text and image-derived content from PDFs.

**Tasks completed:**
- [x] Implement `extract_text()` using PyMuPDF — page-by-page text, chunked at 500 chars with 50-char overlap
- [x] Implement `extract_images()` using PyMuPDF `get_pixmap()` at 150 DPI — renders pages as PNG without Poppler
- [x] Implement `describe_image()` — sends each page PNG to Moondream via Ollama, gets detailed textual description
- [x] Implement `process_pdf()` — orchestrates all three stages with progress callbacks
- [x] Implement `store_in_chroma()` — batch-encodes all chunks with MiniLM, upserts to ChromaDB
- [x] Add CLI entry-point for headless ingestion

**Key challenge solved: VRAM constraint**  
Moondream requires ~2 GB VRAM. Llama 3 requires ~5 GB VRAM. They cannot co-exist on a 6–8 GB consumer GPU. Solution:
- `ollama.chat(..., keep_alive=0)` — Ollama evicts the model after each call
- `gc.collect()` + `torch.cuda.empty_cache()` between VLM calls
- `time.sleep(0.5)` gives Ollama's runner time to release the GPU context

---

### Phase 2 — RAG Query Engine

**Goal:** Answer natural language questions using retrieved chunks.

**Tasks completed:**
- [x] Implement `retrieve()` — embeds query, fetches top-6 chunks from ChromaDB, supports `where=` scoping
- [x] Implement `generate_answer()` — builds a structured RAG prompt, calls Llama 3 via Ollama
- [x] Add `_flush_vram()` before Llama 3 call — ensures Moondream has been evicted
- [x] Handle `CUDA OOM` errors gracefully — return user-friendly message instead of stack trace
- [x] Source metadata returned alongside answer — enables inline image citations in UI
- [x] CLI entry-point for quick testing

**RAG prompt structure:**
```
SYSTEM: You are a Senior AI Document Analyst. Answer using ONLY the context provided
(searching in: <scope>). If visual descriptions are present, treat them as factual
evidence. Be precise and cite page numbers where relevant.

CONTEXT:
[Page 1 – TEXT]
<chunk text>
---
[Page 1 – IMAGE]
[Visual Description – Page 1]: <Moondream output>
---
...

USER: <question>
```

---

### Phase 3 — Streamlit UI

**Goal:** Build a premium, production-quality UI.

**Tasks completed:**
- [x] Dark theme CSS — `#0d0d0f` background, Inter font, glassmorphism cards
- [x] Hero title with purple-to-cyan gradient (`#7c3aed` → `#06b6d4`)
- [x] Sidebar: DB stats cards, document library with per-doc deletion, query scope selector
- [x] Upload widget + progress bar with real-time status messages
- [x] Chat interface: user bubble (right-aligned, purple gradient) + AI bubble (glassmorphism)
- [x] Empty state with suggestion chips
- [x] Source image references shown inline below AI answer
- [x] Active scope banner when scoped to a single document
- [x] Model badges (llama3, moondream, all-MiniLM-L6-v2)
- [x] VRAM flush after ingestion (`_unload_vision_model()`) before first query

**UI design decisions:**
- `max-width: 1100px` main content area — prevents overly wide text on large monitors
- Message bubbles use `margin-left: 15%` (user) and `margin-right: 5%` (AI) to feel natural
- `backdrop-filter: blur(10px)` on AI bubble — glassmorphism depth
- Progress bar uses same purple-cyan gradient as hero

---

### Phase 4 — Multi-document Library

**Goal:** Support many documents simultaneously, with per-document operations.

**Tasks completed:**
- [x] `list_documents()` — groups chunks by `source` metadata, returns name + chunk count
- [x] `delete_document()` — deletes all chunks where `source == filename`
- [x] `clear_collection()` — bulk-deletes all chunks
- [x] Per-document 🗑️ button in sidebar
- [x] Scope selector dynamically populated from document library
- [x] Scope resets to "All Documents" when the scoped document is deleted
- [x] `last_ingested` session state — shows "Last added: filename.pdf" in sidebar

---

### Phase 5 — Polish & Hardening

**Tasks completed:**
- [x] Suppress noisy library logs (sentence_transformers, chromadb, httpx, urllib3)
- [x] `TOKENIZERS_PARALLELISM=false` env var — suppresses tokenizer fork warning
- [x] `pdf2image` kept in requirements (legacy compatibility) but PyMuPDF used for all rendering
- [x] `python310.exe` excluded from git via `.gitignore`
- [x] `chroma_db/`, `temp_uploads/`, `temp_images/` excluded from git
- [x] ChromaDB lazy initialization — `get_collection()` only called when needed
- [x] All `try/except` blocks around Ollama calls — robust error messages to UI
- [x] `gc.collect()` after every query — prevent memory accumulation in long sessions

---

## 6. File-by-File Summary

### `resources.py`
- Defines all configurable constants (`PERSIST_PATH`, `EMBED_MODEL_ID`, `TEXT_LLM`, `VISION_LLM`)
- Lazy singleton getters: `get_embed_model()`, `get_collection()`
- Collection helpers: `get_db_stats()`, `list_documents()`, `delete_document()`, `clear_collection()`
- Silences noisy library logs at module import time

### `ingest.py`
- `extract_text(pdf_path)` → list of text chunk dicts with metadata
- `extract_images(pdf_path)` → list of image path dicts, pages rendered at 150 DPI
- `describe_image(image_path)` → string description from Moondream
- `process_pdf(pdf_path, progress_callback)` → combined chunk list (text + image-derived)
- `store_in_chroma(chunks)` → encodes and upserts to ChromaDB, returns count
- `_unload_vision_model()` → force-evicts Moondream from VRAM
- CLI entry-point: `python ingest.py <path-to-pdf>`

### `query_engine.py`
- `retrieve(query, k, source_filter)` → (docs, metadatas) from ChromaDB
- `generate_answer(query, source_filter)` → (answer_text, metadatas) from Llama 3
- `_flush_vram()` → VRAM cleanup before loading Llama 3
- CLI entry-point: `python query_engine.py "your question"`

### `app.py`
- Page config, global CSS injection (dark theme, gradients, chat bubbles)
- Session state: `messages`, `last_ingested`, `scope`
- Sidebar: stats, library, scope picker, upload, clear-all, model badges
- Main area: hero, scope banner, chat history renderer, chat input handler
- Calls `process_pdf()` + `store_in_chroma()` on upload
- Calls `generate_answer()` for each user message

---

## 7. Known Limitations & Future Work

### Current Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Sequential VLM processing | Slow ingestion for large PDFs (1 page/10–30 s) | Process smaller documents; use CLI during off-hours |
| No streaming LLM output | Answer appears all at once | Acceptable for current use case |
| No table extraction | Tables treated as text or images only | Moondream's visual description partially compensates |
| Fixed chunk size | Not adaptive to document structure | Sufficient for most documents |
| Single collection | All documents in one ChromaDB collection | Per-document deletion via metadata filter works fine |

### Planned Improvements

- [ ] **Streaming responses** — Pipe Ollama streaming to Streamlit's `st.write_stream`
- [ ] **Async ingestion** — Process multiple pages in parallel (requires VRAM upgrade)
- [ ] **Table extraction** — Dedicated table parser using `pdfplumber` or `camelot`
- [ ] **Multi-language support** — Swap MiniLM for multilingual-E5
- [ ] **Export chat** — Save session as PDF or Markdown
- [ ] **Authentication** — Password-protect the Streamlit interface
- [ ] **Document similarity** — Show which documents are most related to a query

---

## 8. Testing Checklist

### Ingestion
- [x] Single-page PDF ingests cleanly
- [x] Multi-page PDF (20+ pages) ingests without VRAM error
- [x] PDF with mostly images ingests and produces image descriptions
- [x] Duplicate ingestion (same file twice) doesn't crash (ChromaDB upsert handles it)
- [x] Progress bar updates correctly through all stages

### Query
- [x] Text-only question answered correctly
- [x] Question about a chart/image answered using Moondream description
- [x] Scoped query ignores content from other documents
- [x] Empty library returns appropriate warning
- [x] CUDA OOM error returns user-friendly message

### UI
- [x] Document library updates immediately after ingestion
- [x] Per-document deletion removes correct chunks
- [x] Clear All removes all chunks and resets scope
- [x] Scope dropdown shows correct options
- [x] Chat history persists within a session
- [x] Source images displayed inline with correct page numbers

---

## 9. Development Timeline

| Date | Milestone |
|---|---|
| Week 1 | Project ideation, technology stack research, Ollama setup |
| Week 1 | `resources.py` — shared singleton pattern established |
| Week 1–2 | `ingest.py` — text extraction + PyMuPDF image rendering |
| Week 2 | Moondream integration, VRAM management strategy |
| Week 2 | `query_engine.py` — retrieval + Llama 3 RAG pipeline |
| Week 2–3 | `app.py` — initial Streamlit UI |
| Week 3 | Multi-document library (list, delete, scope) |
| Week 3 | Premium CSS theme — dark mode, gradients, glassmorphism |
| Week 3 | GPU memory hardening, error handling |
| Week 3 | Chat input visibility fix (black text on light input) |
| Week 3 | Final polish, `.gitignore`, README, CHANGELOG |

---

*Implementation plan written: March 2026*  
*Project status: v1.0.0 — stable*
