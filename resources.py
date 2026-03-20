"""
resources.py — Shared singleton for embed model + ChromaDB.
Import from here in all modules so models load exactly once.
"""
import logging
import warnings
import os

# ── Silence noisy library logs ─────────────────────────────────────────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # suppress tokenizer fork warning

import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_PATH    = "./chroma_db"
COLLECTION_NAME = "multimodal_docs"
EMBED_MODEL_ID  = "all-MiniLM-L6-v2"
TEXT_LLM        = "llama3"
VISION_LLM      = "moondream"

# ── Lazy singletons ────────────────────────────────────────────────────────────
_embed_model  = None
_chroma_client = None
_collection   = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_ID, device='cpu')
    return _embed_model


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=PERSIST_PATH)
        _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def get_db_stats() -> dict:
    col = get_collection()
    count = col.count()
    return {"total_chunks": count}


def clear_collection() -> int:
    col = get_collection()
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)


def list_documents() -> list[dict]:
    """Return list of {name, chunks} dicts for every unique source document."""
    col = get_collection()
    if col.count() == 0:
        return []
    results = col.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"name": k, "chunks": v} for k, v in sorted(counts.items())]


def delete_document(filename: str) -> int:
    """Delete all chunks belonging to a specific document. Returns count removed."""
    col = get_collection()
    results = col.get(where={"source": filename})
    if results["ids"]:
        col.delete(ids=results["ids"])
    return len(results["ids"])
