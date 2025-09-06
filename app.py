{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
from pathlib import Path\
from typing import List\
\
import streamlit as st\
from langchain_community.document_loaders import (\
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredHTMLLoader,\
)\
from langchain.text_splitter import RecursiveCharacterTextSplitter\
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI\
from langchain_community.vectorstores import FAISS\
from langchain.chains import RetrievalQA\
from langchain.prompts import PromptTemplate\
\
# -------------------------\
# CONFIG\
# -------------------------\
BASE_DATA_DIR = Path("data")     # per-user files & indexes live here\
INDEX_SUBDIR   = "rag_index"\
EMBED_MODEL    = "models/embedding-001"\
LLM_MODEL      = "gemini-1.5-flash"\
RETRIEVER_K    = 8\
CHUNK_SIZE     = 800\
CHUNK_OVERLAP  = 150\
\
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)\
\
# -------------------------\
# Helpers\
# -------------------------\
def get_user_folder(username: str) -> Path:\
    uname = "".join(c for c in username if c.isalnum() or c in ("_", "-")).strip() or "anonymous"\
    folder = BASE_DATA_DIR / uname\
    folder.mkdir(parents=True, exist_ok=True)\
    return folder\
\
def _get_loader_for_path(fp: Path):\
    ext = fp.suffix.lower()\
    if ext == ".pdf": return PyPDFLoader\
    if ext in [".txt", ".md"]: return TextLoader\
    if ext == ".csv": return CSVLoader\
    if ext == ".docx": return Docx2txtLoader\
    if ext in [".html", ".htm"]: return UnstructuredHTMLLoader\
    return None\
\
def load_documents_from_folder(folder: Path) -> List:\
    docs = []\
    patterns = ["*.pdf","*.txt","*.md","*.csv","*.docx","*.html","*.htm"]\
    for pattern in patterns:\
        for fp in sorted(folder.rglob(pattern)):\
            Loader = _get_loader_for_path(fp)\
            if not Loader: \
                continue\
            try:\
                if Loader in (CSVLoader, TextLoader):\
                    loader = Loader(str(fp), encoding="utf-8")\
                else:\
                    loader = Loader(str(fp))\
                file_docs = loader.load()\
                for d in file_docs:\
                    d.metadata = getattr(d, "metadata", \{\}) or \{\}\
                    d.metadata["source"] = fp.name\
                docs.extend(file_docs)\
            except Exception as e:\
                st.warning(f"Skipped \{fp.name\}: \{e\}")\
    return docs\
\
def build_faiss_index(data_folder: Path, index_folder: Path):\
    docs = load_documents_from_folder(data_folder)\
    if not docs:\
        raise ValueError("No supported files found to index.")\
    splitter = RecursiveCharacterTextSplitter(\
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,\
        separators=["\\n\\n","\\n"," ",""]\
    )\
    chunks = splitter.split_documents(docs)\
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)\
    vs = FAISS.from_documents(chunks, embeddings)\
    index_folder.mkdir(parents=True, exist_ok=True)\
    vs.save_local(str(index_folder))\
    return vs\
\
def load_or_build_index_for_user(user_folder: Path):\
    index_path = user_folder / INDEX_SUBDIR\
    if index_path.exists() and any(index_path.iterdir()):\
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)\
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)\
    return build_faiss_index(user_folder, index_path)\
\
def create_qa_chain(vectorstore):\
    retriever = vectorstore.as_retriever(search_kwargs=\{"k": RETRIEVER_K\})\
    prompt = PromptTemplate(\
        input_variables=["question", "context"],\
        template=(\
            "You are an expert assistant for clinical trial data standards. "\
            "Use only the provided context to answer. "\
            "If the answer isn't in the context, say \\"I don't know from provided docs.\\" "\
            "Be concise and use bullet points when helpful.\\n\\n"\
            "Question: \{question\}\\n"\
            "Context:\\n\{context\}\\n\\n"\
            "Answer:"\
        ),\
    )\
    return RetrievalQA.from_chain_type(\
        llm=ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, max_output_tokens=1024),\
        retriever=retriever,\
        return_source_documents=True,\
        chain_type_kwargs=\{"prompt": prompt\},\
    )\
\
# -------------------------\
# UI\
# -------------------------\
st.set_page_config(page_title="Clinical Docs Search", layout="wide")\
st.title("\uc0\u55357 \u56538  Clinical Docs Search (per-user RAG)")\
\
st.sidebar.header("User & Files")\
username = st.sidebar.text_input("Your username (team name or email)", value="guest")\
user_folder = get_user_folder(username)\
\
uploaded = st.sidebar.file_uploader(\
    "Upload files (PDF, DOCX, TXT, CSV, MD, HTML)",\
    type=["pdf","docx","txt","csv","md","html"], accept_multiple_files=True\
)\
if uploaded:\
    for f in uploaded:\
        (user_folder / f.name).write_bytes(f.getbuffer())\
    st.sidebar.success(f"Saved \{len(uploaded)\} file(s) to \{user_folder\}")\
\
if st.sidebar.button("\uc0\u55357 \u56615  Build / Rebuild Index"):\
    try:\
        with st.spinner("Indexing\'85 this may take a few minutes for large files."):\
            build_faiss_index(user_folder, user_folder / INDEX_SUBDIR)\
        st.sidebar.success("Index built \uc0\u9989 ")\
    except Exception as e:\
        st.sidebar.error(f"Index build failed: \{e\}")\
\
st.sidebar.markdown("### Your files")\
files = sorted([p.name for p in user_folder.rglob("*") if p.is_file()])\
st.sidebar.write("\\n".join(f"- \{f\}" for f in files) if files else "No files yet.")\
\
st.header("\uc0\u55357 \u56590  Search")\
q = st.text_area("Ask a question about your uploaded docs", height=120)\
\
if st.button("Search"):\
    if not q.strip():\
        st.warning("Type a question first.")\
    elif not os.environ.get("GOOGLE_API_KEY"):\
        st.error("Missing GOOGLE_API_KEY \'97 set as an environment variable or Streamlit secret.")\
    else:\
        with st.spinner("Loading index (or building if missing)\'85"):\
            try:\
                db = load_or_build_index_for_user(user_folder)\
            except Exception as e:\
                db = None\
                st.error(f"Error loading index: \{e\}")\
        if db:\
            qa = create_qa_chain(db)\
            with st.spinner("Retrieving and generating answer\'85"):\
                out = qa.invoke(\{"query": q\})\
            st.subheader("Answer")\
            st.write(out.get("result","").strip())\
\
            st.subheader("Sources (unique)")\
            uniq = list(\{d.metadata.get("source","unknown") for d in out.get("source_documents",[])\})\
            if uniq:\
                for s in uniq: st.write("-", s)\
            else:\
                st.write("No sources returned.")\
\
            st.subheader("Evidence snippets")\
            for i, d in enumerate(out.get("source_documents", [])[:6], 1):\
                st.markdown(f"**\{i\}. \{d.metadata.get('source','unknown')\}** \'97 \{d.page_content[:500].replace('\\n',' ')\}\'85")\
}