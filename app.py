# app.py
import os
import asyncio
from pathlib import Path
from typing import List
import pickle
from math import sqrt

import numpy as np
import streamlit as st

# Ensure an asyncio event loop exists for the current thread (Streamlit-related fix)
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

# LangChain loaders and helpers
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# embeddings / LLM
from langchain_google_genai import ChatGoogleGenerativeAI


# LangChain chain & prompt
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever  # add near other imports
from sentence_transformers import SentenceTransformer



# -------------------------
# CONFIG
# -------------------------
BASE_DATA_DIR = Path("data")
INDEX_SUBDIR = "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
RETRIEVER_K = 8
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

from langchain_huggingface import HuggingFaceEmbeddings


# files inside index folder
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.pkl"

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helpers: file loaders / splitting
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

def load_documents_from_folder(folder: Path) -> List[Document]:
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
                try:
                    st.info(f"Loaded {len(file_docs)} docs from {fp.name}")
                except Exception:
                    pass
            except Exception as e:
                try:
                    st.warning(f"Skipped {fp.name}: {e}")
                except Exception:
                    pass
    return docs

# -------------------------
# Simple vector store (numpy + pickle)
# -------------------------
def persist_simple_index(index_folder: Path, vectors: np.ndarray, docs: List[Document]):
    index_folder.mkdir(parents=True, exist_ok=True)
    np.save(index_folder / VECTORS_FILE, vectors)
    meta = [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {})} for d in docs]
    with open(index_folder / META_FILE, "wb") as f:
        pickle.dump(meta, f)

def load_simple_index(index_folder: Path):
    vpath = index_folder / VECTORS_FILE
    mpath = index_folder / META_FILE
    if not vpath.exists() or not mpath.exists():
        return None, None
    vectors = np.load(vpath)
    with open(mpath, "rb") as f:
        meta = pickle.load(f)
    docs = []
    for m in meta:
        docs.append(Document(page_content=m["page_content"], metadata=m.get("metadata", {})))
    return vectors, docs

def cosine_sim_matrix(vecs: np.ndarray, qvec: np.ndarray):
    # vecs: (n, d), qvec: (d,)
    # compute cosine similarities fast
    denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qvec) + 1e-12)) + 1e-12
    sims = np.dot(vecs, qvec) / denom
    sims = np.nan_to_num(sims)
    return sims

class SimpleRetriever:
    def __init__(self, vectors: np.ndarray, docs: List[Document], embeddings, k=8):
        self.vectors = vectors
        self.docs = docs
        self.embeddings = embeddings
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        # embed the query
        try:
            if hasattr(self.embeddings, "embed_query"):
                qvec = np.array(self.embeddings.embed_query(query))
            else:
                qvec = np.array(self.embeddings.embed_documents([query]))[0]
        except Exception:
            # fallback to embed_documents in loop
            qvec = np.array(self.embeddings.embed_documents([query]))[0]

        sims = cosine_sim_matrix(self.vectors, qvec)
        topk = sims.argsort()[::-1][: self.k]
        docs_out = [self.docs[i] for i in topk]
        return docs_out

# Adapter so our SimpleRetriever looks like a LangChain BaseRetriever
# Adapter that is pydantic-friendly for LangChain
# --- Replace your current SimpleRetrieverAdapter with this updated version ---
from langchain.schema import BaseRetriever
import numpy as np

class SimpleRetrieverAdapter(BaseRetriever):
    """
    Adapter that wraps SimpleRetriever and:
     - is pydantic-friendly (model_config)
     - proxies unknown attrs to the inner simple retriever
     - provides async wrapper and an optional scored-retrieval method
     - exposes `tags` and `metadata` attributes to satisfy callers
    """
    model_config = {"extra": "allow"}

    def __init__(self, simple_retriever):
        # store the wrapped retriever without pydantic validation
        object.__setattr__(self, "simple", simple_retriever)
        # expose a tags field (empty by default)
        object.__setattr__(self, "tags", [])
        # expose metadata field (empty dict by default)
        object.__setattr__(self, "metadata", {})

    def __getattr__(self, name):
        # proxy unknown attributes/methods to the wrapped retriever
        try:
            return getattr(self.simple, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def get_relevant_documents(self, query: str):
        return self.simple.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)

    def get_relevant_documents_with_score(self, query: str):
        """
        Optional helper returning list of (Document, score).
        If the wrapped SimpleRetriever has vectors/docs/embeddings we compute cosine scores.
        Otherwise, fall back to returning (doc, 1.0).
        """
        if hasattr(self.simple, "vectors") and hasattr(self.simple, "docs") and hasattr(self.simple, "embeddings"):
            # embed query
            try:
                embeddings = self.simple.embeddings
                if hasattr(embeddings, "embed_query"):
                    qvec = np.array(embeddings.embed_query(query))
                else:
                    qvec = np.array(embeddings.embed_documents([query]))[0]
            except Exception:
                qvec = np.array(embeddings.embed_documents([query]))[0]

            vecs = self.simple.vectors  # numpy array
            denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qvec) + 1e-12)) + 1e-12
            sims = np.dot(vecs, qvec) / denom
            sims = np.nan_to_num(sims)
            topk_idxs = sims.argsort()[::-1][: self.simple.k if hasattr(self.simple, "k") else sims.shape[0]]
            return [(self.simple.docs[i], float(sims[i])) for i in topk_idxs]
        else:
            docs = self.simple.get_relevant_documents(query)
            return [(d, 1.0) for d in docs]



# -------------------------
# Build / load simple index
# -------------------------
def build_simple_index(data_folder: Path, index_folder: Path):
    docs = load_documents_from_folder(data_folder)
    if not docs:
        raise ValueError("No supported files found to index.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                             separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    texts = [d.page_content for d in chunks]
    # embed documents (may be expensive)
    try:
        vecs = embeddings.embed_documents(texts)
    except Exception:
        vecs = []
        for t in texts:
            vecs.append(embeddings.embed_documents([t])[0])

    vectors = np.array(vecs)
    persist_simple_index(index_folder, vectors, chunks)
    retriever = SimpleRetriever(vectors=vectors, docs=chunks, embeddings=embeddings, k=RETRIEVER_K)
    return retriever

def load_or_build_simple_index_for_user(user_folder: Path):
    index_path = user_folder / INDEX_SUBDIR
    vectors, docs = load_simple_index(index_path)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if vectors is not None and docs is not None:
        return SimpleRetriever(vectors=vectors, docs=docs, embeddings=embeddings, k=RETRIEVER_K)
    # else build
    return build_simple_index(user_folder, index_path)

# -------------------------
# LLM prompt / QA builder
# -------------------------
PROMPT_TEMPLATE_STR = (
    "You are an expert assistant for clinical trial data standards. Use only the provided context to answer.\n"
    "If the answer isn't in the context, say 'I don't know from provided docs.' Be concise and use bullet points when helpful.\n\n"
    "Question: {question}\nContext:\n{context}\n\nAnswer:"
)
prompt_template = PromptTemplate(input_variables=["question", "context"], template=PROMPT_TEMPLATE_STR)

def create_qa_from_retriever(retriever):
    """
    Accepts our SimpleRetriever (or SimpleRetrieverAdapter) and returns a LangChain RetrievalQA.
    We wrap the simple retriever in SimpleRetrieverAdapter so pydantic validation succeeds.
    """
    wrapped = SimpleRetrieverAdapter(retriever)
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, max_output_tokens=1024),
        retriever=wrapped,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa



# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Clinical Docs Search", layout="wide")
st.title("📚 Clinical Docs Search (simple vector store)")

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

if st.sidebar.button("🔧 Build / Rebuild Index"):
    try:
        with st.spinner("Indexing… this may take a few minutes for large files."):
            build_simple_index(user_folder, user_folder / INDEX_SUBDIR)
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
            if not index_path.exists() or not any(index_path.iterdir()):
                with st.spinner("Index not found — building index now (this may take a few minutes)..."):
                    try:
                        build_simple_index(user_folder, index_path)
                        st.success("Index built successfully.")
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")
                        st.markdown("**Files found (for debugging):**")
                        for fp in user_files:
                            st.write("-", fp.name)
                        # show a short sample from first file for debugging
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

            # load retriever and run QA
            try:
                retriever = load_or_build_simple_index_for_user(user_folder)
            except Exception as e:
                st.error(f"Error loading index: {e}")
                retriever = None

            if retriever:
                qa = create_qa_from_retriever(retriever)
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
