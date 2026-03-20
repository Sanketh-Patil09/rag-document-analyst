# 🗄️ Build Your Own Vector Database — Step-by-Step Tutorial

> Learn how to go from raw documents to a fully searchable, semantic vector database — the core of any RAG (Retrieval-Augmented Generation) system.

---

## What You'll Build

By the end of this guide you'll have:

- A **local ChromaDB vector store** loaded with your own documents
- A **semantic search engine** that finds relevant content using meaning, not keywords
- A ready-to-use foundation for any RAG or AI Q&A application

---

## How a Vector Database Works

```
Your Document
     │
     ▼
 Text Chunks          "The revenue grew by 23% in Q3…"
     │
     ▼
 Embedding Model  →   [0.12, -0.45, 0.87, … 384 numbers]
 (all-MiniLM-L6-v2)
     │
     ▼
 Vector Database  →   Stored with metadata (page, source, type)
 (ChromaDB)
     │
     ▼
 Query Embedding  →   Your question → vector
     │
     ▼
 Similarity Search →  Top-K most similar chunks returned
```

**Key concept:** Two semantically similar sentences will have vectors that are close together in space — even if they share no words. This is what makes semantic search so powerful.

---

## Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |

---

## Step 1 — Create a Project Folder

```bash
mkdir my-vector-db
cd my-vector-db
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## Step 2 — Install Dependencies

```bash
pip install chromadb sentence-transformers pymupdf
```

| Package | Purpose |
|---|---|
| `chromadb` | The vector database (local, no server needed) |
| `sentence-transformers` | Converts text to embeddings |
| `pymupdf` | Reads and extracts text from PDF files |

---

## Step 3 — Understand the Core Concept: Embeddings

Before writing code, let's see embeddings in action.

Create `explore_embeddings.py`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The quarterly revenue increased by 23%.",
    "Sales grew significantly in Q3.",
    "The cat sat on the mat.",
]

embeddings = model.encode(sentences)

print(f"Each embedding has {len(embeddings[0])} dimensions\n")

# Compute similarity between first two (should be HIGH)
from sklearn.metrics.pairwise import cosine_similarity
sim_12 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_13 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"Similarity: sentence 1 vs 2 = {sim_12:.3f}  ← semantically similar")
print(f"Similarity: sentence 1 vs 3 = {sim_13:.3f}  ← unrelated")
```

Run it:
```bash
pip install scikit-learn   # just for this demo
python explore_embeddings.py
```

Expected output:
```
Each embedding has 384 dimensions

Similarity: sentence 1 vs 2 = 0.821  ← semantically similar
Similarity: sentence 1 vs 3 = 0.104  ← unrelated
```

> ✅ This is exactly how vector search works — your query gets turned into a vector and the database finds the closest stored vectors.

---

## Step 4 — Set Up ChromaDB

Create `setup_db.py`:

```python
import chromadb

# Create a persistent client — data saved to ./chroma_db on disk
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection (like a table in SQL)
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

print(f"Collection '{collection.name}' is ready.")
print(f"Documents stored: {collection.count()}")
```

Run it:
```bash
python setup_db.py
```

Output:
```
Collection 'my_documents' is ready.
Documents stored: 0
```

> 📁 A `chroma_db/` folder is now created on disk. Your data will persist here between runs.

---

## Step 5 — Chunk Your Text

Raw documents are too long to embed in one shot. We split them into **overlapping chunks** so no important information is cut off at a boundary.

Create `chunker.py`:

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.
    
    chunk_size: max characters per chunk
    overlap: how many characters each chunk shares with the previous one
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # slide forward (with overlap)
    return chunks


# ── Test it ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "word " * 300   # 1500 characters
    chunks = chunk_text(sample)
    print(f"Text length : {len(sample)} characters")
    print(f"Chunk count : {len(chunks)}")
    print(f"Chunk sizes : {[len(c) for c in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0][:80]}…")
```

Run it:
```bash
python chunker.py
```

---

## Step 6 — Ingest a PDF into the Vector Database

Now we put it all together. Create `ingest.py`:

```python
"""
ingest.py — Load a PDF, chunk it, embed it, store it in ChromaDB
"""
import uuid
import fitz   # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# ── Configuration ─────────────────────────────────────────────────────────────
PDF_PATH        = "your_document.pdf"   # ← change this
PERSIST_PATH    = "./chroma_db"
COLLECTION_NAME = "my_documents"
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract per-page text from a PDF and split into chunks."""
    doc = fitz.open(pdf_path)
    filename = pdf_path.split("/")[-1].split("\\")[-1]
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if not page_text.strip():
            continue   # skip blank pages

        for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            if chunk.strip():
                chunks.append({
                    "id":      str(uuid.uuid4()),
                    "text":    chunk,
                    "metadata": {
                        "source": filename,
                        "page":   page_num,
                    }
                })

    doc.close()
    print(f"  ✔ Extracted {len(chunks)} chunks from '{filename}'")
    return chunks


def embed_and_store(chunks: list[dict]):
    """Embed chunks and upsert into ChromaDB."""
    print(f"  ⏳ Loading embedding model '{EMBED_MODEL}'…")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    texts      = [c["text"]     for c in chunks]
    ids        = [c["id"]       for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]

    print(f"  ⏳ Embedding {len(texts)} chunks…")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client     = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"  ⏳ Upserting into ChromaDB…")
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"\n  ✅ Done! Total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    print(f"\n📄 Ingesting: {PDF_PATH}")
    chunks = extract_text_from_pdf(PDF_PATH)
    embed_and_store(chunks)
```

Run it (replace `your_document.pdf` with your actual file):
```bash
python ingest.py
```

Output:
```
📄 Ingesting: annual_report.pdf
  ✔ Extracted 87 chunks from 'annual_report.pdf'
  ⏳ Loading embedding model 'all-MiniLM-L6-v2'…
  ⏳ Embedding 87 chunks…
  ⏳ Upserting into ChromaDB…

  ✅ Done! Total chunks in DB: 87
```

---

## Step 7 — Search Your Vector Database

Create `search.py`:

```python
"""
search.py — Semantic search over your ChromaDB vector store
"""
import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_PATH    = "./chroma_db"
COLLECTION_NAME = "my_documents"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 5   # number of results to return


def search(query: str, top_k: int = TOP_K, source_filter: str = None):
    """
    Search the vector database for chunks semantically similar to `query`.
    
    Args:
        query:         Your natural language question
        top_k:         How many results to return
        source_filter: (optional) restrict search to one document filename
    
    Returns:
        List of (text, metadata, score) tuples
    """
    model      = SentenceTransformer(EMBED_MODEL, device="cpu")
    client     = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("⚠️  The database is empty. Run ingest.py first.")
        return []

    # Embed the query
    query_embedding = model.encode([query]).tolist()

    # Optional: filter to a single document
    where = {"source": source_filter} if source_filter else None

    # Query the database
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Package results
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist   # ChromaDB returns cosine distance; convert to similarity
        hits.append((doc, meta, similarity))

    return hits


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the main findings?"

    print(f"\n🔍 Query: \"{query}\"\n")
    hits = search(query)

    if not hits:
        print("No results found.")
    else:
        for i, (text, meta, score) in enumerate(hits, start=1):
            print(f"─── Result {i} │ Score: {score:.3f} │ {meta['source']} p.{meta['page']} ───")
            print(text[:300] + ("…" if len(text) > 300 else ""))
            print()
```

Run it:
```bash
python search.py "What is the revenue growth?"
```

---

## Step 8 — Manage Your Vector Database

Create `manage_db.py`:

```python
"""
manage_db.py — Inspect, filter, and delete from your ChromaDB
"""
import chromadb

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_documents")


# ── View statistics ───────────────────────────────────────────────────────────
def show_stats():
    total = collection.count()
    print(f"\n📊 Database Statistics")
    print(f"   Total chunks: {total}")
    if total == 0:
        return
    
    # Count per document
    all_data = collection.get(include=["metadatas"])
    counts = {}
    for meta in all_data["metadatas"]:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    
    print(f"\n   Documents:")
    for name, count in sorted(counts.items()):
        print(f"   • {name}: {count} chunks")


# ── Delete a specific document ────────────────────────────────────────────────
def delete_document(filename: str):
    results = collection.get(where={"source": filename})
    if not results["ids"]:
        print(f"⚠️  '{filename}' not found in database.")
        return
    collection.delete(ids=results["ids"])
    print(f"🗑️  Deleted {len(results['ids'])} chunks for '{filename}'")


# ── Clear everything ──────────────────────────────────────────────────────────
def clear_all():
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    print(f"🗑️  Cleared {len(all_ids)} total chunks from the database.")


# ── Run from command line ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"

    if cmd == "stats":
        show_stats()
    elif cmd == "delete" and len(sys.argv) > 2:
        delete_document(sys.argv[2])
    elif cmd == "clear":
        clear_all()
    else:
        print("Usage:")
        print("  python manage_db.py stats")
        print("  python manage_db.py delete <filename.pdf>")
        print("  python manage_db.py clear")
```

Usage:
```bash
python manage_db.py stats
python manage_db.py delete annual_report.pdf
python manage_db.py clear
```

---

## Step 9 — Ingest Plain Text Files Too

Add this function to `ingest.py` to support `.txt` files alongside PDFs:

```python
def extract_text_from_txt(txt_path: str) -> list[dict]:
    """Ingest a plain text file."""
    filename = txt_path.split("/")[-1].split("\\")[-1]
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []
    for i, chunk in enumerate(chunk_text(content)):
        if chunk.strip():
            chunks.append({
                "id":      str(uuid.uuid4()),
                "text":    chunk,
                "metadata": {"source": filename, "page": i + 1},
            })

    print(f"  ✔ Extracted {len(chunks)} chunks from '{filename}'")
    return chunks
```

---

## Step 10 — Add to This RAG Project

Once your vector database is built, you can plug it directly into this project by:

1. Setting `PERSIST_PATH` in `resources.py` to point to your `chroma_db/` folder
2. Making sure `COLLECTION_NAME` matches the collection you created above
3. Running `streamlit run app.py` — your pre-built database will be available immediately

```python
# In resources.py — point to your existing vector DB
PERSIST_PATH    = "./chroma_db"        # same folder you used above
COLLECTION_NAME = "my_documents"       # match your collection name
```

> 💡 You can also build the vector DB using the tutorial scripts and then upload additional documents through the Streamlit UI — they'll all go into the same ChromaDB collection.

---

## Final Project Structure

```
my-vector-db/
├── explore_embeddings.py   # Step 3: understand embeddings
├── setup_db.py             # Step 4: initialise ChromaDB
├── chunker.py              # Step 5: text chunking utility
├── ingest.py               # Step 6: PDF → chunks → DB
├── search.py               # Step 7: semantic search
├── manage_db.py            # Step 8: stats, delete, clear
├── chroma_db/              # Persistent vector store (auto-created)
│   ├── chroma.sqlite3
│   └── ...
└── your_document.pdf       # Your input document(s)
```

---

## Quick Reference

| Task | Command |
|---|---|
| Ingest a PDF | `python ingest.py` (set `PDF_PATH` first) |
| Search the DB | `python search.py "your question"` |
| View DB stats | `python manage_db.py stats` |
| Delete a document | `python manage_db.py delete report.pdf` |
| Clear everything | `python manage_db.py clear` |

---

## Key Concepts Summary

| Term | What It Means |
|---|---|
| **Embedding** | A list of numbers representing the semantic meaning of text |
| **Chunk** | A small piece of a larger document (e.g. 500 characters) |
| **Overlap** | Shared characters between adjacent chunks — prevents info loss at boundaries |
| **Vector Search** | Finding chunks whose embeddings are closest to your query's embedding |
| **Cosine Similarity** | The angle between two vectors — 1.0 = identical meaning, 0.0 = unrelated |
| **Collection** | A named group of vectors in ChromaDB (like a table in SQL) |
| **Persistent Client** | ChromaDB saves data to disk — survives restarts |
| **Metadata** | Extra info stored alongside each chunk (filename, page number, etc.) |
| **Top-K** | How many results to retrieve — typically 5–10 for RAG |
