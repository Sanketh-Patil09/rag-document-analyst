"""
app.py — Multi-modal RAG Document Analyst
Premium dark-themed Streamlit UI with multi-document library
"""
import streamlit as st
import os
import gc
from resources import get_db_stats, clear_collection, list_documents, delete_document, PERSIST_PATH
from query_engine import generate_answer
from ingest import process_pdf, store_in_chroma, _unload_vision_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0f;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }

.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* ── Hero ── */
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.2rem; letter-spacing: -0.5px;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 1.8rem; }

/* ── Badges ── */
.badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.2rem; }
.badge {
    background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.35);
    border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; color: #a78bfa; font-weight: 500;
}

/* ── Chat messages ── */
.msg-user {
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(6,182,212,0.15));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 16px 16px 4px 16px; padding: 1rem 1.2rem;
    margin: 0.5rem 0; margin-left: 15%;
}
.msg-assistant {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px); border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem; margin: 0.5rem 0; margin-right: 5%;
}
.msg-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
.lbl-user { color: #a78bfa; }
.lbl-ai   { color: #06b6d4; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 1rem; color: #334155; }
.empty-state h3 { color: #475569; font-weight: 500; margin-bottom: 0.5rem; }
.suggestion-chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 1rem; }
.chip {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 6px 16px; font-size: 0.82rem; color: #64748b;
}

/* ── Source image caption ── */
.src-label { font-size: 0.75rem; color: #64748b; text-align: center; margin-top: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,15,20,0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Stats card ── */
.stat-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 14px 16px; margin: 8px 0;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #7c3aed; }
.stat-label { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

/* ── Document card ── */
.doc-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 10px 12px; margin: 6px 0;
    display: flex; align-items: center; justify-content: space-between;
}
.doc-name { font-size: 0.82rem; color: #cbd5e1; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 160px; }
.doc-chunks { font-size: 0.7rem; color: #475569; }

/* ── Scope selector ── */
.scope-label { font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.5rem 1.2rem; width: 100%; transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

div[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    border-radius: 14px !important;
    color: #111111 !important;
}

div[data-testid="stProgress"] > div { background-color: rgba(124,58,237,0.2) !important; }
div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#7c3aed,#06b6d4) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_ingested" not in st.session_state:
    st.session_state.last_ingested = None
if "scope" not in st.session_state:
    st.session_state.scope = "📚 All Documents"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 RAG Analyst")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

    # ── DB Stats ──────────────────────────────────────────────────────────────
    stats = get_db_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{stats["total_chunks"]}</div>'
            f'<div class="stat-label">Chunks Stored</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        icon  = "🟢" if stats["total_chunks"] > 0 else "🔴"
        label = "Ready" if stats["total_chunks"] > 0 else "Empty"
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{icon}</div>'
            f'<div class="stat-label">DB {label}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Document Library ──────────────────────────────────────────────────────
    st.markdown("#### 📂 Document Library")
    docs = list_documents()

    if not docs:
        st.markdown("<small style='color:#475569'>No documents ingested yet.</small>", unsafe_allow_html=True)
    else:
        for doc in docs:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f'<div class="doc-card">'
                    f'  <div>'
                    f'    <div class="doc-name" title="{doc["name"]}">📄 {doc["name"]}</div>'
                    f'    <div class="doc-chunks">{doc["chunks"]} chunks</div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c2:
                if st.button("🗑️", key=f"del_{doc['name']}", help=f"Remove {doc['name']}"):
                    removed = delete_document(doc["name"])
                    st.toast(f"Removed {removed} chunks from '{doc['name']}'", icon="🗑️")
                    # If deleted doc was the active scope, reset
                    if st.session_state.scope == doc["name"]:
                        st.session_state.scope = "📚 All Documents"
                    st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Query Scope Selector ──────────────────────────────────────────────────
    st.markdown("#### 🎯 Query Scope")
    scope_options = ["📚 All Documents"] + [d["name"] for d in docs]
    selected_scope = st.selectbox(
        "Search in:",
        options=scope_options,
        index=scope_options.index(st.session_state.scope) if st.session_state.scope in scope_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.scope = selected_scope

    if selected_scope != "📚 All Documents":
        st.markdown(
            f"<small style='color:#7c3aed'>🔍 Scoped to: <b>{selected_scope}</b></small>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<small style='color:#475569'>🔍 Searching across all documents</small>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("#### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        save_dir  = "temp_uploads"
        os.makedirs(save_dir, exist_ok=True)
        temp_path = os.path.join(save_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.caption(f"📎 **{uploaded_file.name}** ({round(uploaded_file.size/1024, 1)} KB)")

        if st.button("⚡ Process & Add to Library"):
            progress_bar = st.progress(0.0)
            status_text  = st.empty()

            def ui_progress(msg, frac):
                progress_bar.progress(frac)
                status_text.markdown(f"<small style='color:#64748b'>{msg}</small>", unsafe_allow_html=True)

            chunks = process_pdf(temp_path, progress_callback=ui_progress)
            added  = store_in_chroma(chunks)

            # ── Critical: flush moondream from VRAM so llama3 can load ──
            ui_progress("🧹 Freeing GPU memory…", 0.97)
            _unload_vision_model()
            gc.collect()

            progress_bar.progress(1.0)
            status_text.empty()
            st.success(f"✅ {added} chunks added! GPU memory freed — you can now ask questions.")
            st.session_state.last_ingested = uploaded_file.name
            st.rerun()

    if st.session_state.last_ingested:
        st.caption(f"Last added: **{st.session_state.last_ingested}**")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Clear All ─────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Entire Library"):
        cleared = clear_collection()
        st.session_state.last_ingested = None
        st.session_state.scope = "📚 All Documents"
        st.warning(f"Library cleared ({cleared} chunks removed).")
        st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Model badges ──────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Active Models")
    st.markdown("""
<div class="badge-row">
  <span class="badge">🤖 llama3</span>
  <span class="badge">👁️ moondream</span>
  <span class="badge">📐 all-MiniLM-L6-v2</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Multi-modal Document Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by Llama3 · Moondream · ChromaDB — 100% local, 100% private</div>', unsafe_allow_html=True)

# ── Active scope banner ────────────────────────────────────────────────────────
if st.session_state.scope != "📚 All Documents":
    st.info(f"🎯 **Scoped mode** — Answers will only use content from: `{st.session_state.scope}`")

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="empty-state">
  <h3>Ask anything about your documents</h3>
  <p>Upload PDFs in the sidebar, pick a scope, then ask:</p>
  <div class="suggestion-chips">
    <span class="chip">📊 Describe the charts on page 1</span>
    <span class="chip">📝 Summarise the key findings</span>
    <span class="chip">🔢 List all key numbers mentioned</span>
    <span class="chip">🖼️ What images are in the document?</span>
  </div>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    role = message["role"]
    if role == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-label lbl-user">You</div>{message["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="msg-assistant"><div class="msg-label lbl-ai">🧠 AI Analyst</div>{message["content"]}</div>',
            unsafe_allow_html=True
        )
        if message.get("images"):
            img_cols = st.columns(min(len(message["images"]), 3))
            for i, img_path in enumerate(message["images"]):
                if os.path.exists(img_path):
                    with img_cols[i]:
                        st.image(img_path, use_container_width=True)
                        page = message.get("image_pages", [None]*10)[i]
                        st.markdown(f'<div class="src-label">📄 Page {page}</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your document(s)…"):
    stats = get_db_stats()
    if stats["total_chunks"] == 0:
        st.warning("⚠️ Library is empty. Please upload and process a PDF first.")
    else:
        active_scope = st.session_state.scope
        source_filter = None if active_scope == "📚 All Documents" else active_scope

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f'<div class="msg-user"><div class="msg-label lbl-user">You</div>{prompt}</div>',
            unsafe_allow_html=True
        )

        scope_label = f"in **{source_filter}**" if source_filter else "across **all documents**"
        with st.spinner(f"🔍 Searching {scope_label}…"):
            answer, metadatas = generate_answer(prompt, source_filter=source_filter)
            gc.collect()

        # Collect source images
        source_images, image_pages = [], []
        for meta in metadatas:
            if meta.get("type") == "image":
                img_path = meta.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    source_images.append(img_path)
                    image_pages.append(meta.get("page"))

        st.markdown(
            f'<div class="msg-assistant"><div class="msg-label lbl-ai">🧠 AI Analyst</div>{answer}</div>',
            unsafe_allow_html=True
        )

        if source_images:
            st.markdown("<small style='color:#475569;margin-left:4px'>📌 Source References</small>", unsafe_allow_html=True)
            img_cols = st.columns(min(len(source_images), 3))
            for i, img_path in enumerate(source_images[:3]):
                with img_cols[i]:
                    st.image(img_path, use_container_width=True)
                    st.markdown(f'<div class="src-label">📄 Page {image_pages[i]}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     answer,
            "images":      source_images[:3],
            "image_pages": image_pages[:3],
        })