# app.py
import os
import asyncio
from pathlib import Path
from typing import List

import streamlit as st

# Ensure an asyncio event loop exists for the current thread (Streamlit-related fix)
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embeddings and llm
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Chroma vectorstore
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------
# CONFIG
# -------------------------
BASE_DATA_DIR = Path("data")
INDEX_SUBDIR = "rag_index"
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"
RETRIEVER_K = 8
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def get_user_folder(username: str) -> Path:
    uname = "".join(c for c in username if c.isalnum() or c in ("_", "-")).strip() or "anonymous"
    folder = BASE_DATA_DIR / uname
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def _get_loader_for_path(fp: Path):
    ext = fp.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader
    if ext in [".txt", ".md"]:
        return TextLoader
    if ext == ".csv":
        return CSVLoader
    if ext == ".docx":
        return Docx2txtLoader
    if ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader
    return None

def load_documents_from_folder(folder: Path) -> List:
    docs = []
    patterns = ["*.pdf", "*.txt", "*.md", "*.csv", "*.docx", "*.html", "*.htm"]
    for pattern in patterns:
        for fp in sorted(folder.rglob(pattern)):
            Loader = _get_loader_for_path(fp)
            if not Loader:
                continue
            try:
                if Loader in (CSVLoader, TextLoader):
                    loader = Loader(str(fp), encoding="utf-8")
                else:
                    loader = Loader(str(fp))
                file_docs = loader.load()
                # attach source metadata (filename only)
                for d in file_docs:
                    d.metadata = getattr(d, "metadata", {}) or {}
                    d.metadata["source"] = fp.name
                docs.extend(file_docs)
                # Log to Streamlit (only if running in Streamlit context)
                try:
                    st.info(f"Loaded {len(file_docs)} docs from {fp.name}")
                except Exception:
                    pass
            except Exception as e:
                # show a friendly warning in UI and continue
                try:
                    st.warning(f"Skipped {fp.name}: {e}")
                except Exception:
                    pass
    return docs

def build_chroma_index(data_folder: Path, index_folder: Path):
    """
    Build a Chroma index. Try persistent Chroma (disk). If that fails
    (eg. unsupported sqlite on host), fall back to in-memory Chroma.
    """
    docs = load_documents_from_folder(data_folder)
    if not docs:
        raise ValueError("No supported files found to index.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    persist_directory = str(index_folder)

    # Try persistent Chroma
    try:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
        try:
            # persist if available
            vs.persist()
        except Exception:
            pass
        try:
            st.info("Persistent Chroma index built.")
        except Exception:
            pass
        return vs
    except Exception as e:
        # fallback to in-memory Chroma for demos
        try:
            st.warning(f"Persistent Chroma failed ({e}). Falling back to in-memory index (non-persistent).")
        except Exception:
            pass
        try:
            vs = Chroma.from_documents(documents=chunks, embedding=embeddings)
            return vs
        except Exception as e2:
            raise RuntimeError(f"Failed to build Chroma index. Persistent error: {e}; in-memory error: {e2}")

def load_or_build_index_for_user(user_folder: Path):
    """
    Try to load a persistent Chroma DB if present; if not present, attempt to build.
    If load fails due to system sqlite, fall back to in-memory.
    """
    index_path = user_folder / INDEX_SUBDIR

    # If a persistent index folder exists, try loading it
    if index_path.exists() and any(index_path.iterdir()):
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        try:
            vs = Chroma(persist_directory=str(index_path), embedding_function=embeddings)
            try:
                st.info("Loaded existing persistent Chroma index.")
            except Exception:
                pass
            return vs
        except Exception as e:
            try:
                st.warning(f"Failed to load persistent Chroma index ({e}). Rebuilding (may fall back to in-memory).")
            except Exception:
                pass
            # fall through to rebuilding

    # Build (will try persistent, then fallback to in-memory)
    return build_chroma_index(user_folder, index_path)

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are an expert assistant for clinical trial data standards. "
            "Use only the provided context to answer. "
            "If the answer isn't in the context, say \"I don't know from provided docs.\" "
            "Be concise and use bullet points when helpful.\n\n"
            "Question: {question}\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
    )
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, max_output_tokens=1024),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Clinical Docs Search", layout="wide")
st.title("📚 Clinical Docs Search (per-user RAG)")

st.sidebar.header("User & Files")
username = st.sidebar.text_input("Your username (team name or email)", value="guest")
user_folder = get_user_folder(username)

uploaded = st.sidebar.file_uploader(
    "Upload files (PDF, DOCX, TXT, CSV, MD, HTML)",
    type=["pdf", "docx", "txt", "csv", "md", "html"],
    accept_multiple_files=True,
)
if uploaded:
    saved = 0
    for f in uploaded:
        try:
            (user_folder / f.name).write_bytes(f.getbuffer())
            saved += 1
        except Exception as e:
            st.warning(f"Failed to save {f.name}: {e}")
    st.sidebar.success(f"Saved {saved} file(s) to {user_folder}")

# Manual build button
if st.sidebar.button("🔧 Build / Rebuild Index"):
    try:
        with st.spinner("Indexing… this may take a few minutes for large files."):
            build_chroma_index(user_folder, user_folder / INDEX_SUBDIR)
        st.sidebar.success("Index built ✅")
    except Exception as e:
        st.sidebar.error(f"Index build failed: {e}")

st.sidebar.markdown("### Your files")
files = sorted([p.name for p in user_folder.rglob("*") if p.is_file()])
if files:
    for fname in files:
        st.sidebar.write("-", fname)
else:
    st.sidebar.write("No files yet. Upload on the left and then click Build / Rebuild Index.")

st.header("🔎 Search")
q = st.text_area("Ask a question about your uploaded docs", height=120)

# Improved Search handling with auto-build when files exist
if st.button("Search"):
    if not q.strip():
        st.warning("Type a question first.")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY — set as an environment variable or Streamlit secret.")
    else:
        user_files = sorted([p for p in user_folder.rglob("*") if p.is_file()])
        if not user_files:
            st.error(
                "No files found in your folder. Upload files using the sidebar uploader and then click "
                "'Build / Rebuild Index' (or upload and the app will build automatically)."
            )
        else:
            st.info(f"Found {len(user_files)} file(s) in your folder. Checking index...")
            index_path = user_folder / INDEX_SUBDIR
            # If index missing, auto-build
            if not index_path.exists() or not any(index_path.iterdir()):
                with st.spinner("Index not found — building index now (this may take a few minutes)..."):
                    try:
                        build_chroma_index(user_folder, index_path)
                        st.success("Index built successfully.")
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")
                        st.markdown("**Files found (for debugging):**")
                        for fp in user_files:
                            st.write("-", fp.name)
                        # try to show sample extract of first file
                        try:
                            sample_fp = user_files[0]
                            Loader = PyPDFLoader if sample_fp.suffix.lower() == ".pdf" else TextLoader
                            loader = Loader(str(sample_fp)) if Loader is PyPDFLoader else Loader(str(sample_fp), encoding="utf-8")
                            sample_docs = loader.load()
                            if sample_docs:
                                st.markdown("**Sample extract (first doc):**")
                                st.code(sample_docs[0].page_content[:800])
                        except Exception:
                            pass
                        st.stop()

            # load the index and run QA
            try:
                db = load_or_build_index_for_user(user_folder)
            except Exception as e:
                st.error(f"Error loading index: {e}")
                db = None

            if db:
                qa = create_qa_chain(db)
                with st.spinner("Retrieving and generating answer…"):
                    try:
                        out = qa.invoke({"query": q})
                    except Exception as e:
                        st.error(f"LLM/query error: {e}")
                        out = None
                if out:
                    st.subheader("Answer")
                    st.write(out.get("result", "").strip())

                    st.subheader("Sources (unique)")
                    uniq = list({d.metadata.get("source", "unknown") for d in out.get("source_documents", [])})
                    if uniq:
                        for s in uniq:
                            st.write("-", s)
                    else:
                        st.write("No sources returned.")

                    st.subheader("Evidence snippets")
                    for i, d in enumerate(out.get("source_documents", [])[:6], 1):
                        st.markdown(f"**{i}. {d.metadata.get('source','unknown')}** — {d.page_content[:500].replace('\\n', ' ')}…")
