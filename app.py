# app.py - ENHANCED VERSION
# Added: Caching, Error Handling, Metrics, Cost Tracking, Performance Monitoring

import os
import asyncio
from pathlib import Path
from typing import List
import pickle
from math import sqrt
import time
from datetime import datetime
import hashlib

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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

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

# files inside index folder
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.pkl"

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# NEW: METRICS & COST TRACKING
# ========================================
class MetricsTracker:
    """Track system performance and costs"""
    def __init__(self):
        self.queries = []
        self.total_queries = 0
        self.cache_hits = 0
        self.total_tokens = 0
        self.errors = 0
        
    def log_query(self, query: str, response_time: float, cached: bool = False, tokens: int = 0, error: bool = False):
        self.total_queries += 1
        if cached:
            self.cache_hits += 1
        if error:
            self.errors += 1
        self.total_tokens += tokens
        
        self.queries.append({
            "query": query[:100],
            "response_time": response_time,
            "cached": cached,
            "tokens": tokens,
            "timestamp": datetime.now(),
            "error": error
        })
    
    def get_stats(self):
        if self.total_queries == 0:
            return {
                "total_queries": 0,
                "cache_hit_rate": "0%",
                "avg_response_time": 0,
                "error_rate": "0%",
                "total_tokens": 0,
                "estimated_cost": "$0.00"
            }
        
        cache_rate = (self.cache_hits / self.total_queries) * 100
        error_rate = (self.errors / self.total_queries) * 100
        
        non_cached = [q for q in self.queries if not q["cached"]]
        avg_time = sum(q["response_time"] for q in non_cached) / len(non_cached) if non_cached else 0
        
        # Gemini Flash pricing: ~$0.35 per 1M tokens (input + output)
        estimated_cost = (self.total_tokens / 1_000_000) * 0.35
        
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "avg_response_time": f"{avg_time:.2f}s",
            "error_rate": f"{error_rate:.1f}%",
            "total_tokens": self.total_tokens,
            "estimated_cost": f"${estimated_cost:.4f}"
        }

# Initialize metrics in session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = MetricsTracker()


# ========================================
# NEW: QUERY CACHE
# ========================================
class QueryCache:
    """Simple query cache to reduce API calls"""
    def __init__(self, max_size=50, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
    
    def _get_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str):
        key = self._get_key(query)
        if key in self.cache:
            result, timestamp = self.cache[key]
            # Check if expired
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, result):
        key = self._get_key(query)
        # If cache is full, remove oldest
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (result, time.time())
    
    def get_stats(self):
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage": f"{(len(self.cache) / self.max_size) * 100:.1f}%"
        }

# Initialize cache in session state
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = QueryCache(max_size=50, ttl_seconds=3600)


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
                st.info(f"✅ Loaded {len(file_docs)} docs from {fp.name}")
            except Exception as e:
                st.warning(f"⚠️ Skipped {fp.name}: {e}")
    return docs


# -------------------------
# NEW: CACHED VECTOR STORE LOADING
# -------------------------
@st.cache_resource
def load_embeddings_model():
    """Cache the embeddings model to avoid reloading"""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


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
        try:
            if hasattr(self.embeddings, "embed_query"):
                qvec = np.array(self.embeddings.embed_query(query))
            else:
                qvec = np.array(self.embeddings.embed_documents([query]))[0]
        except Exception:
            qvec = np.array(self.embeddings.embed_documents([query]))[0]

        sims = cosine_sim_matrix(self.vectors, qvec)
        topk = sims.argsort()[::-1][: self.k]
        docs_out = [self.docs[i] for i in topk]
        return docs_out

class SimpleRetrieverAdapter(BaseRetriever):
    model_config = {"extra": "allow"}

    def __init__(self, simple_retriever):
        object.__setattr__(self, "simple", simple_retriever)
        object.__setattr__(self, "tags", [])
        object.__setattr__(self, "metadata", {})

    def __getattr__(self, name):
        try:
            return getattr(self.simple, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def get_relevant_documents(self, query: str):
        return self.simple.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)

    def get_relevant_documents_with_score(self, query: str):
        if hasattr(self.simple, "vectors") and hasattr(self.simple, "docs") and hasattr(self.simple, "embeddings"):
            try:
                embeddings = self.simple.embeddings
                if hasattr(embeddings, "embed_query"):
                    qvec = np.array(embeddings.embed_query(query))
                else:
                    qvec = np.array(embeddings.embed_documents([query]))[0]
            except Exception:
                qvec = np.array(embeddings.embed_documents([query]))[0]

            vecs = self.simple.vectors
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
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    st.info(f"📊 Created {len(chunks)} chunks from {len(docs)} documents")

    embeddings = load_embeddings_model()

    texts = [d.page_content for d in chunks]
    
    # Show progress for embedding
    progress_bar = st.progress(0)
    vecs = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_vecs = embeddings.embed_documents(batch)
            vecs.extend(batch_vecs)
        except Exception:
            for t in batch:
                vecs.append(embeddings.embed_documents([t])[0])
        progress_bar.progress(min((i + batch_size) / len(texts), 1.0))

    vectors = np.array(vecs)
    persist_simple_index(index_folder, vectors, chunks)
    
    st.success(f"✅ Index saved with {len(chunks)} chunks")
    
    retriever = SimpleRetriever(vectors=vectors, docs=chunks, embeddings=embeddings, k=RETRIEVER_K)
    return retriever

def load_or_build_simple_index_for_user(user_folder: Path):
    index_path = user_folder / INDEX_SUBDIR
    vectors, docs = load_simple_index(index_path)
    embeddings = load_embeddings_model()
    
    if vectors is not None and docs is not None:
        st.info(f"📚 Loaded existing index with {len(docs)} chunks")
        return SimpleRetriever(vectors=vectors, docs=docs, embeddings=embeddings, k=RETRIEVER_K)
    
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
    wrapped = SimpleRetrieverAdapter(retriever)
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, max_output_tokens=1024),
        retriever=wrapped,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa


# ========================================
# NEW: QUERY WITH CACHING & ERROR HANDLING
# ========================================
def query_with_features(qa_chain, query: str):
    """
    Query with caching, error handling, and metrics tracking
    """
    start_time = time.time()
    cache = st.session_state.query_cache
    metrics = st.session_state.metrics
    
    # Check cache
    cached_result = cache.get(query)
    if cached_result:
        elapsed = time.time() - start_time
        metrics.log_query(query, elapsed, cached=True)
        st.success("🎯 Cache hit! Instant response.")
        return cached_result, True, elapsed
    
    # Not in cache, query LLM with retry logic
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = qa_chain.invoke({"query": query})
            elapsed = time.time() - start_time
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(query.split()) + len(result.get("result", "").split()) * 1.3
            
            metrics.log_query(query, elapsed, cached=False, tokens=int(estimated_tokens))
            
            # Cache the result
            cache.set(query, result)
            
            return result, False, elapsed
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                st.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                elapsed = time.time() - start_time
                metrics.log_query(query, elapsed, error=True)
                st.error(f"❌ All retries failed: {last_error}")
                return None, False, elapsed
    
    return None, False, time.time() - start_time


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Clinical Docs Search", layout="wide")

# ========================================
# NEW: HEADER WITH METRICS
# ========================================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 Clinical Docs Search")
with col2:
    stats = st.session_state.metrics.get_stats()
    st.metric("Total Queries", stats["total_queries"])

# ========================================
# NEW: METRICS DASHBOARD IN SIDEBAR
# ========================================
st.sidebar.header("📊 System Metrics")
stats = st.session_state.metrics.get_stats()

col1, col2 = st.sidebar.columns(2)
col1.metric("Cache Hit Rate", stats["cache_hit_rate"])
col2.metric("Error Rate", stats["error_rate"])

col1.metric("Avg Response", stats["avg_response_time"])
col2.metric("Total Tokens", f"{stats['total_tokens']:,}")

st.sidebar.metric("Estimated Cost", stats["estimated_cost"])

# Cache stats
cache_stats = st.session_state.query_cache.get_stats()
st.sidebar.metric("Cache Usage", f"{cache_stats['size']}/{cache_stats['max_size']}")

st.sidebar.markdown("---")

# ========================================
# EXISTING UI (Enhanced)
# ========================================
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
    st.sidebar.success(f"✅ Saved {saved} file(s)")

if st.sidebar.button("🔧 Build / Rebuild Index"):
    try:
        with st.spinner("Indexing… this may take a few minutes for large files."):
            build_simple_index(user_folder, user_folder / INDEX_SUBDIR)
        st.sidebar.success("✅ Index built successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Index build failed: {e}")

st.sidebar.markdown("### Your files")
files = sorted([p.name for p in user_folder.rglob("*") if p.is_file() and p.name not in [VECTORS_FILE, META_FILE]])
if files:
    for fname in files:
        st.sidebar.write("📄", fname)
else:
    st.sidebar.write("No files yet. Upload files and click Build Index.")

# ========================================
# SEARCH INTERFACE (Enhanced)
# ========================================
st.header("🔎 Search")
q = st.text_area("Ask a question about your uploaded docs", height=120)

# Add example questions
with st.expander("💡 Example Questions"):
    st.markdown("""
    - What are the main findings of the study?
    - What are the inclusion criteria?
    - What adverse events were reported?
    - Summarize the methodology
    """)

if st.button("🔍 Search", type="primary"):
    if not q.strip():
        st.warning("⚠️ Please enter a question first.")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("❌ Missing GOOGLE_API_KEY — set as an environment variable or Streamlit secret.")
    else:
        user_files = sorted([p for p in user_folder.rglob("*") if p.is_file() and p.name not in [VECTORS_FILE, META_FILE]])
        
        if not user_files:
            st.error("❌ No files found. Please upload files and build the index first.")
        else:
            index_path = user_folder / INDEX_SUBDIR
            
            if not index_path.exists() or not any(index_path.iterdir()):
                with st.spinner("📦 Index not found — building now..."):
                    try:
                        build_simple_index(user_folder, index_path)
                        st.success("✅ Index built successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to build index: {e}")
                        st.stop()

            # Load retriever and query
            try:
                retriever = load_or_build_simple_index_for_user(user_folder)
            except Exception as e:
                st.error(f"❌ Error loading index: {e}")
                retriever = None

            if retriever:
                qa = create_qa_from_retriever(retriever)
                
                with st.spinner("🔍 Retrieving and generating answer…"):
                    result, was_cached, elapsed = query_with_features(qa, q)
                
                if result:
                    # Display answer with metrics
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.subheader("✨ Answer")
                    with col2:
                        st.metric("Response Time", f"{elapsed:.2f}s")
                    with col3:
                        st.metric("Source", "🎯 Cache" if was_cached else "🤖 LLM")
                    
                    st.markdown(result.get("result", "").strip())

                    # Sources section
                    st.subheader("📚 Sources")
                    uniq = list({d.metadata.get("source", "unknown") for d in result.get("source_documents", [])})
                    if uniq:
                        for s in uniq:
                            st.write("📄", s)
                    
                    # Evidence snippets in expander
                    with st.expander("🔍 View Evidence Snippets"):
                        for i, d in enumerate(result.get("source_documents", [])[:6], 1):
                            st.markdown(f"**{i}. {d.metadata.get('source','unknown')}**")
                            st.text(d.page_content[:400].replace('\n', ' '))
                            st.markdown("---")
                    
                    # User feedback
                    st.markdown("### Was this helpful?")
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("👍 Yes"):
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎 No"):
                            feedback = st.text_input("What could be improved?")
                            if feedback:
                                st.info("Feedback recorded. Thank you!")
