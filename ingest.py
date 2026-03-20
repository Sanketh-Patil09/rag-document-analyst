"""
ingest.py — Multi-modal PDF ingestion pipeline
"""
import os
import uuid
import gc
import time
import fitz  # PyMuPDF
import ollama

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from resources import (
    get_embed_model, get_collection, get_db_stats,
    PERSIST_PATH, VISION_LLM,
)

CHUNK_SIZE    = 500  # characters per text chunk
CHUNK_OVERLAP = 50   # overlap between chunks


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split a long string into overlapping fixed-size chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + size])
        start += size - overlap
    return chunks


# ── Pipeline stages ─────────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> list[dict]:
    """Extract per-page text, split into chunks, and attach metadata."""
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    results = []

    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text()
        if not raw.strip():
            continue
        for chunk in _chunk_text(raw):
            if chunk.strip():
                results.append({
                    "id":      str(uuid.uuid4()),
                    "content": chunk,
                    "metadata": {"page": page_num, "type": "text", "source": filename},
                })

    doc.close()
    return results


def extract_images(pdf_path: str, output_folder: str = "temp_images") -> list[dict]:
    """Render each PDF page to a PNG using PyMuPDF — no Poppler needed."""
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    results = []

    for i, page in enumerate(doc, start=1):
        mat = fitz.Matrix(150 / 72, 150 / 72)   # 150 DPI
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(
            output_folder,
            f"{os.path.splitext(filename)[0]}_page_{i}.png"
        )
        pix.save(img_path)
        results.append({
            "id":   str(uuid.uuid4()),
            "path": img_path,
            "metadata": {"page": i, "type": "image", "source": filename},
        })

    doc.close()
    return results


def describe_image(image_path: str) -> str:
    """Send a page image to the VLM and get a detailed textual description."""
    response = ollama.chat(
        model=VISION_LLM,
        keep_alive=0,   # unload model from VRAM immediately after this call
        messages=[{
            "role":    "user",
            "content": (
                "You are a data analyst. Describe this image in detail, "
                "covering any charts, graphs, tables, key numbers, and insights."
            ),
            "images": [image_path],
        }]
    )
    return response["message"]["content"]


def process_pdf(pdf_path: str, progress_callback=None) -> list[dict]:
    """
    Full ingestion pipeline:
      1. Extract text chunks
      2. Render pages as images
      3. Describe each image with the VLM
    Returns combined chunk list.
    """
    if progress_callback:
        progress_callback("📄 Extracting text…", 0.1)
    text_chunks = extract_text(pdf_path)

    if progress_callback:
        progress_callback("🖼️ Rendering pages as images…", 0.35)
    images = extract_images(pdf_path)

    image_chunks = []
    total = len(images)
    for idx, img in enumerate(images, start=1):
        if progress_callback:
            frac = 0.35 + 0.55 * (idx / total)
            progress_callback(f"🧠 Describing page {idx}/{total} with VLM…", frac)

        description = describe_image(img["path"])

        # ── Aggressive VRAM cleanup between VLM calls ──────────────────────────
        gc.collect()
        if _has_torch:
            torch.cuda.empty_cache()
        time.sleep(0.5)   # give Ollama's runner time to release GPU memory

        image_chunks.append({
            "id":      img["id"],
            "content": f"[Visual Description – Page {img['metadata']['page']}]: {description}",
            "metadata": {
                "page":       img["metadata"]["page"],
                "type":       "image",
                "source":     img["metadata"]["source"],
                "image_path": img["path"],
            },
        })

    return text_chunks + image_chunks


def _unload_vision_model():
    """
    Explicitly tell Ollama to evict moondream from VRAM.
    Call this after ingestion is complete so llama3 can load cleanly.
    """
    try:
        ollama.generate(model=VISION_LLM, prompt="", keep_alive=0)
    except Exception:
        pass  # ignore — the goal is just to trigger unload
    gc.collect()
    if _has_torch:
        torch.cuda.empty_cache()
    time.sleep(1.0)  # give Ollama's server time to release the GPU context


def store_in_chroma(chunks: list[dict]) -> int:
    """Embed and upsert chunks into ChromaDB. Returns number of chunks added."""
    if not chunks:
        return 0

    embed_model = get_embed_model()
    collection  = get_collection()

    texts      = [c["content"]  for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()
    ids        = [c["id"]       for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(chunks)


# ── CLI entry-point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\mrnek\RAG\sample.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    def cli_progress(msg, frac):
        bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
        print(f"[{bar}] {msg}")

    chunks = process_pdf(pdf_path, progress_callback=cli_progress)
    added  = store_in_chroma(chunks)
    print(f"\n✅ Done — {added} chunks stored.")