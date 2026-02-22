#!/usr/bin/env python3
"""
indexer.py — v1.2
Shared indexing logic + correct mixed-file loading
"""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_DIR = Path("knowledge")
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def build_index():
    """Build or rebuild the full FAISS index from knowledge/ and all subfolders."""
    print("🔨 Building FAISS index from knowledge/ and all subfolders...")

    knowledge_dir = KNOWLEDGE_DIR.resolve()
    print(f"→ Scanning: {knowledge_dir}")
    print(f"  Exists: {knowledge_dir.exists()} | Is dir: {knowledge_dir.is_dir()}")

    if not knowledge_dir.exists():
        print("❌ 'knowledge' folder not found! Run: mkdir -p knowledge")
        return None

    # Debug: count files
    pdf_files = list(knowledge_dir.rglob("*.pdf"))
    md_files = list(knowledge_dir.rglob("*.md"))
    txt_files = list(knowledge_dir.rglob("*.txt"))
    csv_files = list(knowledge_dir.rglob("*.csv"))

    print(f"  Found {len(pdf_files)} PDF, {len(md_files)} MD, {len(txt_files)} TXT, {len(csv_files)} CSV")

    if len(pdf_files) + len(md_files) + len(txt_files) + len(csv_files) == 0:
        print("⚠️  No supported files found! Drop .pdf/.md/.txt/.csv into knowledge/")
        return None

    knowledge = str(knowledge_dir)

    pdf_docs = DirectoryLoader(
        knowledge, glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True, use_multithreading=True, recursive=True
    ).load()

    md_docs = DirectoryLoader(
        knowledge, glob="**/*.md", loader_cls=TextLoader,
        show_progress=True, use_multithreading=True, recursive=True
    ).load()

    txt_docs = DirectoryLoader(
        knowledge, glob="**/*.txt", loader_cls=TextLoader,
        show_progress=True, use_multithreading=True, recursive=True
    ).load()

    csv_docs = DirectoryLoader(
        knowledge, glob="**/*.csv", loader_cls=CSVLoader,
        show_progress=True, use_multithreading=True, recursive=True
    ).load()

    all_docs = pdf_docs + md_docs + txt_docs + csv_docs
    print(f"✅ Loaded {len(all_docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print("🎉 Index built successfully!")
    print(f"   Saved to: {FAISS_INDEX_DIR}/")
    return vectorstore


if __name__ == "__main__":
    build_index()
