# app.py - CLEANED & CONSOLIDATED
import os
import asyncio
from pathlib import Path
from typing import List
import pickle
from math import sqrt
import time
from datetime import datetime
import hashlib
import traceback

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

# -------------------------
# LangChain loaders and helpers (robust across versions)
# -------------------------
# Document fallback imports
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        try:
            from langchain.schema import Document
        except Exception:
            class Document:
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

# Text splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None

# Document loaders
_loaded_loaders = False
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
        Docx2txtLoader,
        UnstructuredHTMLLoader,
    )
    _loaded_loaders = True
except Exception:
    try:
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            CSVLoader,
            Docx2txtLoader,
            UnstructuredHTMLLoader,
        )
        _loaded_loaders = True
    except Exception:
        PyPDFLoader = TextLoader = CSVLoader = Docx2txtLoader = UnstructuredHTMLLoader = None

# RetrievalQA import (multiple possible locations)
RetrievalQA = None
try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        try:
            from langchain.chains import RetrievalQA
        except Exception:
            RetrievalQA = None

# PromptTemplate import (try a few locations)
try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain.prompt import PromptTemplate
    except Exception:
        class PromptTemplate:
            def __init__(self, input_variables=None, template=""):
                self.input_variables = input_variables or []
                self.template = template

# BaseRetriever type (adapter uses it)
try:
    from langchain.schema import BaseRetriever
except Exception:
    try:
        from langchain_core.schema import BaseRetriever
    except Exception:
        BaseRetriever = object

# Google Gemini integration (langchain-google-genai package)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# HuggingFace embeddings wrapper
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

# sentence-transformers (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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

VECTORS_FILE = "vectors.npy"
META_FILE = "meta.pkl"

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# METRICS & CACHE
# ========================================
class MetricsTracker:
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
        estimated_cost = (self.total_tokens / 1_000_000) * 0.35
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "avg_response_time": f"{avg_time:.2f}s",
            "error_rate": f"{error_rate:.1f}%",
            "total_tokens": self.total_tokens,
            "estimated_cost": f"${estimated_cost:.4f}"
        }

if 'metrics' not in st.session_state:
    st.session_state.metrics = MetricsTracker()

class QueryCache:
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
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None

    def set(self, query: str, result):
        key = self._get_key(query)
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (result, time.time())

    def get_stats(self):
        return {"size": len(self.cache), "max_size": self.max_size, "usage": f"{(len(self.cache) / self.max_size) * 100:.1f}%"}

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
# Cached embeddings loader
# -------------------------
@st.cache_resource
def load_embeddings_model():
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("HuggingFaceEmbeddings not available — install langchain-huggingface or provide embeddings.")
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# -------------------------
# Simple vector store
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


# -------------------------
# Robust extractor for LLM responses (must be before wrappers)
# -------------------------
def _extract_text_from_llm_response(raw):
    try:
        if raw is None:
            return "", raw
        if isinstance(raw, str):
            return raw, raw
        if isinstance(raw, dict):
            for k in ("result", "text", "content", "message", "output", "response"):
                if k in raw and isinstance(raw[k], str):
                    return raw[k], raw
            if "candidates" in raw and isinstance(raw["candidates"], (list, tuple)) and raw["candidates"]:
                c = raw["candidates"][0]
                if isinstance(c, str):
                    return c, raw
                if isinstance(c, dict) and "content" in c and isinstance(c["content"], str):
                    return c["content"], raw
        if hasattr(raw, "generations"):
            try:
                gens = getattr(raw, "generations")
                if gens:
                    first = gens[0]
                    if isinstance(first, list) and first:
                        g = first[0]
                    else:
                        g = first
                    for attr in ("text", "content", "message"):
                        if hasattr(g, attr):
                            val = getattr(g, attr)
                            if isinstance(val, str) and val:
                                return val, raw
                            if hasattr(val, "content") and isinstance(val.content, str):
                                return val.content, raw
                    if hasattr(g, "text") and isinstance(g.text, str):
                        return g.text, raw
            except Exception:
                pass
        for attr in ("content", "text", "message", "result"):
            if hasattr(raw, attr):
                val = getattr(raw, attr)
                if isinstance(val, str):
                    return val, raw
                if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
                    return val.content, raw
        return str(raw), raw
    except Exception as e:
        return f"<unextractable response: {e}>", raw


# -------------------------
# Adapter class for SimpleRetriever
# -------------------------
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
# Robust fallback QA wrapper (single)
# -------------------------
class SimpleQAWrapper:
    """
    Robust fallback QA wrapper that prefers ChatGoogleGenerativeAI.invoke(str).
    """

    def __init__(self, llm, retriever, prompt_template):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt_template

    def _build_input(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
        if hasattr(self.prompt, "template"):
            prompt_text = self.prompt.template.format(question=query, context=context)
        else:
            prompt_text = f"Question: {query}\nContext:\n{context}"
        return prompt_text, docs

    def _call_llm_variants(self, prompt_text: str):
        last_err = None
        last_raw = None

        # Prefer invoking with a plain string (works for your wrapper)
        inv_fn = getattr(self.llm, "invoke", None)
        if callable(inv_fn):
            try:
                raw = inv_fn(prompt_text)
                last_raw = raw
                if hasattr(raw, "content") and isinstance(getattr(raw, "content"), str):
                    return getattr(raw, "content"), raw, getattr(raw, "source_documents", None) or []
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                last_err = e

        # Try generate with list-of-dicts
        gen_fn = getattr(self.llm, "generate", None)
        if callable(gen_fn):
            try:
                raw = gen_fn([{"content": prompt_text}])
                last_raw = raw
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                last_err = e

        # Other callables
        for name in ("predict", "create", "chat", "respond", "answer"):
            fn = getattr(self.llm, name, None)
            if not callable(fn):
                continue
            try:
                raw = fn(prompt_text)
                last_raw = raw
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                last_err = e
                continue

        raw_type = type(last_raw).__name__ if last_raw is not None else "None"
        raw_preview = repr(last_raw)[:1000] if last_raw is not None else "<no raw captured>"
        raise RuntimeError(f"No callable LLM methods succeeded. last_err: {last_err} | last_raw_type: {raw_type} | last_raw_preview: {raw_preview}")

    def run(self, query: str):
        prompt_text, docs = self._build_input(query)
        text, raw, src_docs = self._call_llm_variants(prompt_text)
        source_documents = src_docs or docs
        return {"result": text, "source_documents": source_documents, "raw": raw}


# -------------------------
# Build / load simple index
# -------------------------
def build_simple_index(data_folder: Path, index_folder: Path):
    docs = load_documents_from_folder(data_folder)
    if not docs:
        raise ValueError("No supported files found to index.")
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError("Text splitter not available — install langchain-text-splitters or compatible langchain.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)
    st.info(f"📊 Created {len(chunks)} chunks from {len(docs)} documents")
    embeddings = load_embeddings_model()
    texts = [d.page_content for d in chunks]
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
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("ChatGoogleGenerativeAI (langchain-google-genai) not available; install it.")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, max_output_tokens=1024)
    try:
        if RetrievalQA is None:
            raise ImportError("RetrievalQA not importable in this environment.")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=wrapped,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
        )
        return qa
    except Exception as e:
        st.warning(f"⚠️ Could not create RetrievalQA chain (falling back to internal wrapper): {e}")
        return SimpleQAWrapper(llm=llm, retriever=wrapped, prompt_template=prompt_template)


# -------------------------
# Chain-safe invoker
# -------------------------
def _call_chain_safe(qa_chain, query: str):
    """
    Try common invocation methods and normalize output:
    returns {"result": str, "source_documents": [Document, ...], "raw": raw}
    Accepts a LangChain RetrievalQA-like object OR our SimpleQAWrapper instance.
    """
    last_err = None
    last_raw = None

    # If it's our fallback wrapper, call its run(query)
    if hasattr(qa_chain, "run") and callable(getattr(qa_chain, "run")) and type(qa_chain).__name__ == "SimpleQAWrapper":
        try:
            return qa_chain.run(query)
        except Exception as e:
            raise RuntimeError(f"Could not invoke QA chain - last error: {e}")

    # Otherwise try common langchain chain call patterns
    call_methods = [("run", lambda fn, q: fn(q)), ("invoke", lambda fn, q: fn(q)), ("__call__", lambda fn, q: fn(q)), ("predict", lambda fn, q: fn(q)), ("generate", lambda fn, q: fn([q]))]
    for name, caller in call_methods:
        fn = getattr(qa_chain, name, None)
        if not callable(fn):
            continue
        try:
            raw = caller(fn, query)
            last_raw = raw
            # if raw is dict
            if isinstance(raw, dict):
                for k in ("result", "text", "content", "message", "output", "response"):
                    if k in raw and isinstance(raw[k], str):
                        return {"result": raw[k], "source_documents": raw.get("source_documents", []) or raw.get("documents", []), "raw": raw}
                if "candidates" in raw and isinstance(raw["candidates"], (list, tuple)) and raw["candidates"]:
                    c0 = raw["candidates"][0]
                    if isinstance(c0, str):
                        return {"result": c0, "source_documents": raw.get("source_documents", []) or raw.get("documents", []), "raw": raw}
                    if isinstance(c0, dict) and "content" in c0 and isinstance(c0["content"], str):
                        return {"result": c0["content"], "source_documents": raw.get("source_documents", []) or raw.get("documents", []), "raw": raw}
            if isinstance(raw, str):
                return {"result": raw, "source_documents": [], "raw": raw}
            # langchain-style generations
            try:
                if hasattr(raw, "generations"):
                    gens = getattr(raw, "generations")
                    if gens:
                        g0 = gens[0]
                        candidate = g0[0] if isinstance(g0, list) and g0 else g0
                        for attr in ("text", "content", "message"):
                            if hasattr(candidate, attr):
                                val = getattr(candidate, attr)
                                if isinstance(val, str):
                                    return {"result": val, "source_documents": getattr(raw, "source_documents", []) or [], "raw": raw}
                                if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
                                    return {"result": getattr(val, "content"), "source_documents": getattr(raw, "source_documents", []) or [], "raw": raw}
            except Exception:
                pass
            # final best-effort by checking attributes
            source_docs = getattr(raw, "source_documents", None) or (raw.get("source_documents") if isinstance(raw, dict) else None) or []
            for attr in ("text", "content", "message", "result"):
                if hasattr(raw, attr):
                    val = getattr(raw, attr)
                    if isinstance(val, str):
                        return {"result": val, "source_documents": source_docs or [], "raw": raw}
                    if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
                        return {"result": getattr(val, "content"), "source_documents": source_docs or [], "raw": raw}
            return {"result": str(raw), "source_documents": source_docs or [], "raw": raw}
        except Exception as e:
            last_err = e
            continue

    debug_raw_repr = repr(last_raw) if last_raw is not None else "<no raw captured>"
    debug_raw_type = type(last_raw).__name__ if last_raw is not None else "None"
    raise RuntimeError(f"Could not invoke QA chain - last error: {last_err} | last_raw_type: {debug_raw_type} | last_raw_preview: {debug_raw_repr[:1000]}")


# -------------------------
# Query wrapper with caching & metrics
# -------------------------
def query_with_features(qa_chain, query: str):
    start_time = time.time()
    cache = st.session_state.query_cache
    metrics = st.session_state.metrics

    cached_result = cache.get(query)
    if cached_result:
        elapsed = time.time() - start_time
        metrics.log_query(query, elapsed, cached=True)
        st.success("🎯 Cache hit! Instant response.")
        return cached_result, True, elapsed

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            result = _call_chain_safe(qa_chain, query)
            elapsed = time.time() - start_time
            estimated_tokens = int(len(query.split()) + len(result.get("result", "").split()) * 1.3)
            metrics.log_query(query, elapsed, cached=False, tokens=int(estimated_tokens))
            cache.set(query, result)
            return result, False, elapsed
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                elapsed = time.time() - start_time
                metrics.log_query(query, elapsed, error=True)
                err_str = str(e)
                debug_hint = ""
                try:
                    if "last_raw_type" in err_str or "last_raw_preview" in err_str:
                        debug_hint = "\n\nDebug info from chain:\n" + err_str
                except Exception:
                    debug_hint = f"\n\nException repr: {repr(e)}"
                st.error(f"❌ All retries failed: {err_str}{debug_hint}")
                st.write("----\n**Debug (copy this and paste in chat):**")
                st.write("Exception:", repr(e))
                try:
                    if "last_raw_preview" in err_str:
                        preview = err_str.split("last_raw_preview: ", 1)[-1]
                        st.code(preview[:5000])
                except Exception:
                    pass
                return None, False, elapsed

    return None, False, time.time() - start_time


# -------------------------
# Streamlit UI (bottom)
# -------------------------
st.set_page_config(page_title="Clinical Docs Search", layout="wide")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 Clinical Docs Search")
with col2:
    stats = st.session_state.metrics.get_stats()
    st.metric("Total Queries", stats["total_queries"])

st.sidebar.header("📊 System Metrics")
stats = st.session_state.metrics.get_stats()
col1, col2 = st.sidebar.columns(2)
col1.metric("Cache Hit Rate", stats["cache_hit_rate"])
col2.metric("Error Rate", stats["error_rate"])
col1.metric("Avg Response", stats["avg_response_time"])
col2.metric("Total Tokens", f"{stats['total_tokens']:,}")
st.sidebar.metric("Estimated Cost", stats["estimated_cost"])
cache_stats = st.session_state.query_cache.get_stats()
st.sidebar.metric("Cache Usage", f"{cache_stats['size']}/{cache_stats['max_size']}")
st.sidebar.markdown("---")

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

st.header("🔎 Search")
q = st.text_area("Ask a question about your uploaded docs", height=120)
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
            try:
                retriever = load_or_build_simple_index_for_user(user_folder)
            except Exception as e:
                st.error(f"❌ Error loading index: {e}")
                retriever = None
            if retriever:
                try:
                    qa = create_qa_from_retriever(retriever)
                except Exception as e:
                    st.error(f"❌ QA chain init failed: {e}")
                    st.stop()
                with st.spinner("🔍 Retrieving and generating answer…"):
                    result, was_cached, elapsed = query_with_features(qa, q)
                if result:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.subheader("✨ Answer")
                    with col2:
                        st.metric("Response Time", f"{elapsed:.2f}s")
                    with col3:
                        st.metric("Source", "🎯 Cache" if was_cached else "🤖 LLM")
                    st.markdown(result.get("result", "").strip() if result.get("result") else "")
                    st.subheader("📚 Sources")
                    uniq = list({d.metadata.get("source", "unknown") for d in result.get("source_documents", [])})
                    if uniq:
                        for s in uniq:
                            st.write("📄", s)
                    with st.expander("🔍 View Evidence Snippets"):
                        for i, d in enumerate(result.get("source_documents", [])[:6], 1):
                            st.markdown(f"**{i}. {d.metadata.get('source','unknown')}**")
                            st.text(d.page_content[:400].replace('\n', ' '))
                            st.markdown("---")
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
