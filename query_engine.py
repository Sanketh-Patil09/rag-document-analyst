"""
query_engine.py — Retrieval + LLM reasoning over ChromaDB
"""
import gc
import time
import ollama
from resources import get_embed_model, get_collection, TEXT_LLM

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

TOP_K = 6  # number of chunks to retrieve per query


def _flush_vram():
    """Free Python objects and GPU memory before loading a new model."""
    gc.collect()
    if _has_torch:
        torch.cuda.empty_cache()
    time.sleep(0.3)   # brief pause so Ollama's runner can release its GPU context


def retrieve(query: str, k: int = TOP_K, source_filter: str | None = None) -> tuple[list[str], list[dict]]:
    """
    Embed the query and return the top-k most relevant chunks.
    Pass source_filter=<filename> to restrict search to one document.
    Returns: (docs, metadatas)
    """
    embed_model = get_embed_model()
    collection  = get_collection()

    count = collection.count()
    if count == 0:
        return [], []

    query_embedding = embed_model.encode([query], show_progress_bar=False).tolist()

    # ChromaDB where= filter for single-document scoping
    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, count),
        where=where,
    )
    return results["documents"][0], results["metadatas"][0]


def generate_answer(query: str, source_filter: str | None = None) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve → build prompt → call Llama3.
    Pass source_filter=<filename> to scope retrieval to one document.
    Returns: (answer_text, metadatas)
    """
    docs, metadatas = retrieve(query, source_filter=source_filter)

    if not docs:
        return (
            "⚠️ The knowledge base is empty. Please upload and process a PDF first.",
            []
        )

    # Build labelled context block
    context = "\n\n---\n\n".join(
        f"[Page {meta.get('page', '?')} – {meta.get('type', 'text').upper()}]\n{doc}"
        for doc, meta in zip(docs, metadatas)
    )

    scope_note = f" (searching only in: {source_filter})" if source_filter else " (searching all documents)"
    system_prompt = (
        "You are a Senior AI Document Analyst. "
        f"Answer the user's question using ONLY the context provided{scope_note}. "
        "If visual descriptions are present, treat them as factual evidence. "
        "Be precise and cite page numbers where relevant.\n\n"
        f"CONTEXT:\n{context}"
    )

    # ── Flush VRAM before loading llama3 (moondream may still hold GPU memory) ──
    _flush_vram()

    try:
        response = ollama.chat(
            model=TEXT_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ]
        )
        return response["message"]["content"], metadatas

    except Exception as e:
        err = str(e)
        if "CUDA" in err or "cuda" in err or "500" in err:
            return (
                "⚠️ **GPU memory error**: The system ran out of VRAM. "
                "This usually happens right after document ingestion. "
                "Please **wait 10–15 seconds** and try your question again — "
                "Ollama needs time to free the GPU memory used by the vision model.",
                metadatas
            )
        return (f"❌ LLM error: {err}", metadatas)


# ── CLI quick-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarise this document."
    answer, metas = generate_answer(query)
    print("\n" + answer)
    imgs = [m.get("image_path") for m in metas if m.get("type") == "image"]
    if imgs:
        print("\nSource images:", imgs)