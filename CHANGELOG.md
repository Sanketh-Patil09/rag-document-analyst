# Changelog

All notable changes to the **Multi-modal RAG Document Analyst** are documented here.

---

## [1.0.0] — 2026-03-20

### Added
- **Multi-modal ingestion pipeline** (`ingest.py`)
  - Text extraction via `PyMuPDF` with configurable chunking (500 chars, 50 overlap)
  - Page rendering at 150 DPI → PNG images (no Poppler dependency)
  - Vision-language model (Moondream) describes every page image
  - Full VRAM flush between VLM calls to prevent OOM errors
- **Semantic retrieval engine** (`query_engine.py`)
  - `all-MiniLM-L6-v2` sentence-transformer embeddings (CPU)
  - ChromaDB top-k (k=6) vector similarity search
  - Per-document scoped search via `where=` filter
  - Llama3 answer generation with structured context prompt
  - Graceful CUDA OOM error handling with user-friendly message
- **Streamlit UI** (`app.py`)
  - Premium dark-themed interface (Inter font, purple-cyan gradient palette)
  - Sidebar: live document library, per-document deletion, DB stats
  - Scoped query selector (all docs vs. single doc)
  - PDF upload + one-click ingest with progress bar
  - Chat interface with user/AI bubble styling
  - Source image references shown inline with AI answers
  - Empty state with suggestion chips
- **Shared resource module** (`resources.py`)
  - Lazy singleton pattern for embed model and ChromaDB client
  - `list_documents()`, `delete_document()`, `clear_collection()` helpers
- **ChromaDB persistence** at `./chroma_db` (survives app restarts)
- **CLI entry-points** in `ingest.py` and `query_engine.py` for headless use
- Initial `requirements.txt`

---

## [Unreleased]

### Planned
- Multi-language document support
- Table extraction as structured data
- Streaming LLM responses
- Export chat session as PDF/Markdown
- Password-protected UI mode
