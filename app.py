# app.py
import os
from pathlib import Path
from typing import List

import streamlit as st

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embeddings / LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Chroma vectorstore (portable for Streamlit Cloud)
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# put this near the top of app.py (right after imports)
import asyncio

def ensure_event_loop():
    try:
        # If there's already a running loop, this is OK
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread — create and set one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()
 
# -------------------------
# CONFIG
# -------------------------
BASE_DATA_DIR = Path("data")     # per-user folders live here
INDEX_SUBDIR   = "rag_index"
EMBED_MODEL    = "models/embedding-001"
LLM_MODEL      = "gemini-1.5-flash"
RETRIEVER_K    = 8
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 150

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
    patterns = ["*.pdf","*.txt","*.md","*.csv","*.docx","*.html","*.htm"]
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
                # show small warning in UI context if called from Streamlit
                try:
                    st.warning(f"Skipped {fp.name}: {e}")
                except Exception:
                    pass
    return docs

def build_chroma_index(data_folder: Path, index_folder: Path):
    docs = load_documents_from_folder(data_folder)
    if not docs:
        raise ValueError("No supported files found to index.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n"," ",""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    persist_directory = str(index_folder)
    # build and persist Chroma DB
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    # .persist() may not be necessary depending on langchain version; call if available
    try:
        vs.persist()
    except Exception:
        pass
    return vs

def load_or_build_index_for_user(user_folder: Path):
    index_path = user_folder / INDEX_SUBDIR
    if index_path.exists() and any(index_path.iterdir()):
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        # load existing chroma persistence
        try:
            vs = Chroma(persist_directory=str(index_path), embedding_function=embeddings)
            return vs
        except Exception as e:
            # fallback to rebuilding if load fails
            st.warning(f"Failed to load existing index (will rebuild): {e}")
            return build_chroma_index(user_folder, index_path)
    else:
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
    type=["pdf","docx","txt","csv","md","html"], accept_multiple_files=True
)
if uploaded:
    for f in uploaded:
        (user_folder / f.name).write_bytes(f.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded)} file(s) to {user_folder}")

if st.sidebar.button("🔧 Build / Rebuild Index"):
    try:
        with st.spinner("Indexing… this may take a few minutes for large files."):
            build_chroma_index(user_folder, user_folder / INDEX_SUBDIR)
        st.sidebar.success("Index built ✅")
    except Exception as e:
        st.sidebar.error(f"Index build failed: {e}")

st.sidebar.markdown("### Your files")
files = sorted([p.name for p in user_folder.rglob("*") if p.is_file()])
st.sidebar.write("\n".join(f"- {f}" for f in files) if files else "No files yet.")

st.header("🔎 Search")
q = st.text_area("Ask a question about your uploaded docs", height=120)

if st.button("Search"):
    if not q.strip():
        st.warning("Type a question first.")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY — set as an environment variable or Streamlit secret.")
    else:
        with st.spinner("Loading index (or building if missing)…"):
            try:
                db = load_or_build_index_for_user(user_folder)
            except Exception as e:
                db = None
                st.error(f"Error loading index: {e}")
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
                st.write(out.get("result","").strip())

                st.subheader("Sources (unique)")
                uniq = list({d.metadata.get("source","unknown") for d in out.get("source_documents",[])})
                if uniq:
                    for s in uniq: st.write("-", s)
                else:
                    st.write("No sources returned.")

                st.subheader("Evidence snippets")
                for i, d in enumerate(out.get("source_documents", [])[:6], 1):
                    st.markdown(f"**{i}. {d.metadata.get('source','unknown')}** — {d.page_content[:500].replace('\\n',' ')}…")
