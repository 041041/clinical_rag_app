# app.py
"""
Streamlit RAG app (ready-to-paste)

Features:
- Upload files (pdf/docx/txt/csv/md/html)
- Build embeddings using Google Generative AI Embeddings
- Simple numpy-backed vector store + metadata persistence
- Manual retrieval + direct LLM call (avoids RetrievalQA recursion issues)
- Safe rendering & truncation to prevent frontend crashes

Before running:
- Set GOOGLE_API_KEY in environment or Streamlit secrets.
- Install required packages (see header comments).
"""

import os
import pickle
import traceback
from pathlib import Path
from typing import List, Any

import numpy as np
import streamlit as st

# LangChain imports (may vary by version)
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
        Docx2txtLoader,
        UnstructuredHTMLLoader,
    )
except Exception:
    # Fallback graceful message — user will see this in UI
    PyPDFLoader = TextLoader = CSVLoader = Docx2txtLoader = UnstructuredHTMLLoader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from langchain.docstore.document import Document
except Exception:
    # very small fallback Document class
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Embeddings and LLM
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except Exception:
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

# ----------------- Configuration -----------------
BASE_DATA_DIR = Path("user_data")  # where user folders and indexes are stored
INDEX_SUBDIR = "rag_index"
EMBED_MODEL = "models/embedding-001"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVER_K = 8
LLM_MODEL = "gemini-1.5-flash"  # change as desired
MAX_CONTEXT_CHARS = 4000
MAX_ANSWER_CHARS = 4000
SHOW_DOCS = 6

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Utility helpers -----------------
def require_env_key(key_name: str):
    if not os.getenv(key_name):
        st.error(f"Environment variable `{key_name}` not found. Set it in Streamlit secrets or env.")
        st.stop()

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

def load_documents_from_folder(folder: Path) -> List[Document]:
    if PyPDFLoader is None:
        raise RuntimeError("Document loaders unavailable. Ensure langchain_community is installed.")
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
                for d in file_docs:
                    d.metadata = getattr(d, "metadata", {}) or {}
                    d.metadata["source"] = fp.name
                docs.extend(file_docs)
            except Exception as e:
                st.warning(f"Skipped {fp.name}: {e}")
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError("Text splitter unavailable. Install langchain properly.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# ----------------- Vector store persistence (numpy + pickle) -----------------
VECTORS_FILENAME = "vectors.npy"
META_FILENAME = "meta.pkl"

def persist_index(index_folder: Path, vectors: np.ndarray, docs: List[Document]):
    index_folder.mkdir(parents=True, exist_ok=True)
    np.save(index_folder / VECTORS_FILENAME, vectors)
    meta = [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {})} for d in docs]
    with open(index_folder / META_FILENAME, "wb") as f:
        pickle.dump(meta, f)

def load_index(index_folder: Path):
    vfile = index_folder / VECTORS_FILENAME
    mfile = index_folder / META_FILENAME
    if not vfile.exists() or not mfile.exists():
        return None, None
    vectors = np.load(vfile)
    with open(mfile, "rb") as f:
        meta = pickle.load(f)
    docs = [Document(page_content=m["page_content"], metadata=m.get("metadata", {})) for m in meta]
    return vectors, docs

# ----------------- Simple Retriever -----------------
class SimpleRetriever:
    """Small synchronous retriever using numpy cosine similarity."""
    def __init__(self, vectors: np.ndarray, docs: List[Document], embeddings, k=RETRIEVER_K):
        self.vectors = vectors
        self.docs = docs
        self.embeddings = embeddings
        self.k = k

    @staticmethod
    def _cosine_sim_matrix(vecs, qvec):
        denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qvec) + 1e-12)) + 1e-12
        sims = np.dot(vecs, qvec) / denom
        return np.nan_to_num(sims)

    def get_relevant_documents(self, query: str):
        if self.vectors is None or len(self.vectors) == 0:
            return []
        # embed query (some embedding wrappers use embed_query)
        try:
            if hasattr(self.embeddings, "embed_query"):
                qvec = np.array(self.embeddings.embed_query(query))
            else:
                qvec = np.array(self.embeddings.embed_documents([query]))[0]
        except Exception:
            # last-resort
            qvec = np.array(self.embeddings.embed_documents([query]))[0]
        sims = self._cosine_sim_matrix(self.vectors, qvec)
        topk_idx = sims.argsort()[::-1][: self.k]
        return [self.docs[i] for i in topk_idx]

# ----------------- Adapter for LangChain compatibility (safe) -----------------
from langchain.schema import BaseRetriever

class SimpleRetrieverAdapter(BaseRetriever):
    """
    Adapter compatible with multiple LangChain versions.
    Implements both protected (_get_relevant_documents) and legacy public methods.
    """
    model_config = {"extra": "allow"}

    def __init__(self, simple_retriever: SimpleRetriever):
        object.__setattr__(self, "simple", simple_retriever)
        object.__setattr__(self, "tags", [])
        object.__setattr__(self, "metadata", {})

    def __getattr__(self, name: str) -> Any:
        try:
            simple = object.__getattribute__(self, "simple")
        except Exception:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
        try:
            return getattr(simple, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def _get_relevant_documents(self, query: str, *args, **kwargs) -> List[Document]:
        return self.simple.get_relevant_documents(query)

    async def _aget_relevant_documents(self, query: str, *args, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, *args, **kwargs)

    def get_relevant_documents(self, query: str, *args, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, *args, **kwargs)

    async def aget_relevant_documents(self, query: str, *args, **kwargs) -> List[Document]:
        return await self._aget_relevant_documents(query, *args, **kwargs)

# ----------------- Core indexing / loading pipeline -----------------
def build_index_for_user(user_folder: Path):
    index_path = user_folder / INDEX_SUBDIR
    # Load docs
    docs = load_documents_from_folder(user_folder)
    if not docs:
        raise ValueError(f"No supported files found in {user_folder}. Upload files first and click 'Build Index'.")
    # Split
    chunks = split_documents(docs)
    # Embeddings
    if GoogleGenerativeAIEmbeddings is None:
        raise RuntimeError("GoogleGenerativeAIEmbeddings not available. Install langchain-google-genai package.")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    texts = [d.page_content for d in chunks]
    vecs = embeddings.embed_documents(texts)
    vectors = np.array(vecs)
    persist_index(index_path, vectors, chunks)
    return vectors, chunks, embeddings

def load_or_build_index_for_user(user_folder: Path):
    index_path = user_folder / INDEX_SUBDIR
    vectors, docs = load_index(index_path)
    if vectors is not None and docs is not None:
        # instantiate embeddings for query-time usage
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError("GoogleGenerativeAIEmbeddings not available.")
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        return vectors, docs, embeddings
    # build
    return build_index_for_user(user_folder)

# ----------------- Manual retrieval + direct LLM call (safe) -----------------
def build_context_from_docs(docs: List[Document], max_chars=MAX_CONTEXT_CHARS):
    pieces = []
    cur = 0
    for d in docs:
        src = (d.metadata.get("source") if getattr(d, "metadata", None) else "unknown")
        txt = d.page_content if hasattr(d, "page_content") else str(d)
        if len(txt) > 1200:
            txt = txt[:1200] + "..."
        block = f"Source: {src}\n\n{txt}\n\n"
        if cur + len(block) > max_chars:
            break
        pieces.append(block)
        cur += len(block)
    return "\n".join(pieces)

def safe_call_llm(prompt: str):
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("ChatGoogleGenerativeAI not available. Install langchain-google-genai package.")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, max_output_tokens=2048)
    # Try different calling styles depending on version
    try:
        out = llm.invoke({"input": prompt})
        if isinstance(out, dict):
            return out.get("result") or out.get("output") or str(out)
        return str(out)
    except Exception:
        try:
            out = llm(prompt)
            return out if isinstance(out, str) else str(out)
        except Exception as e:
            raise e

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Clinical RAG (Safe)", layout="wide")
st.title("📚 Clinical RAG — Safe & Robust")

st.sidebar.header("User & Uploads")
username = st.sidebar.text_input("Username (used for per-user folder)", value="guest")
user_folder = BASE_DATA_DIR / username
user_folder.mkdir(parents=True, exist_ok=True)

uploaded = st.sidebar.file_uploader(
    "Upload files (pdf/docx/txt/csv/md/html)",
    type=["pdf", "docx", "txt", "csv", "md", "html"],
    accept_multiple_files=True,
)
if uploaded:
    saved = 0
    for f in uploaded:
        (user_folder / f.name).write_bytes(f.getbuffer())
        saved += 1
    st.sidebar.success(f"Saved {saved} file(s) to {user_folder}")

st.sidebar.markdown("---")
if st.sidebar.button("Build / Rebuild Index"):
    try:
        require_env_key("GOOGLE_API_KEY")
        with st.spinner("Building index and embeddings (this may take a while)..."):
            vectors, docs, embeddings = build_index_for_user(user_folder)
        st.sidebar.success(f"Index built with {len(docs)} chunks.")
    except Exception as e:
        st.sidebar.error(f"Index build failed: {e}")
        traceback.print_exc()

st.sidebar.markdown("### Files in your folder")
files = sorted([p.name for p in user_folder.rglob("*") if p.is_file()])
if files:
    for f in files:
        st.sidebar.write("-", f)
else:
    st.sidebar.write("No files uploaded yet.")

st.header("Search (manual retrieval + LLM)")

query = st.text_area("Enter your question here", height=120)
k = st.number_input("Top-k chunks to retrieve", min_value=1, max_value=20, value=RETRIEVER_K, step=1)

if st.button("Search"):
    if not query.strip():
        st.warning("Type a question first.")
    else:
        try:
            require_env_key("GOOGLE_API_KEY")
            # Load or build index
            with st.spinner("Loading index and embeddings..."):
                vectors, docs, embeddings = load_or_build_index_for_user(user_folder)
            if vectors is None or docs is None:
                st.error("Index missing. Upload files and click Build Index first.")
            else:
                # create retriever and adapter
                retriever_simple = SimpleRetriever(vectors=vectors, docs=docs, embeddings=embeddings, k=k)
                retriever = SimpleRetrieverAdapter(retriever_simple)

                # manual retrieval
                top_docs = retriever.get_relevant_documents(query)
                # Build context from top-k
                context = build_context_from_docs(top_docs)

                prompt = (
                    "You are an expert assistant. Use ONLY the context below to answer the question. "
                    "If the answer is not present in the context, say 'Not found in provided docs'.\n\n"
                    f"Context:\n{context}\n\nQuestion:\n{query}\n\n"
                    "Answer (short summary then details; cite filenames used):"
                )

                # Call LLM safely
                with st.spinner("Calling LLM..."):
                    try:
                        answer = safe_call_llm(prompt)
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
                        traceback.print_exc()
                        answer = "LLM call failed. Check server logs."

                # Truncate for safe rendering
                if answer is None:
                    answer = ""
                if len(answer) > MAX_ANSWER_CHARS:
                    answer = answer[:MAX_ANSWER_CHARS] + "\n\n...[truncated]..."

                # Display
                with st.expander("Answer"):
                    st.text_area("Answer", value=answer, height=300)

                st.subheader("Top evidence (truncated snippets)")
                for i, d in enumerate(top_docs[:SHOW_DOCS], 1):
                    src = (d.metadata.get("source") if getattr(d, "metadata", None) else "unknown")
                    snippet = d.page_content[:800] + ("..." if len(d.page_content) > 800 else "")
                    st.markdown(f"**{i}. Source:** `{src}`")
                    st.code(snippet)

        except Exception as e:
            st.error(f"Search failed: {e}")
            traceback.print_exc()
