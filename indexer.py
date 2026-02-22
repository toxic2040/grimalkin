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
<<<<<<< Updated upstream
    """Build or rebuild FAISS index — now with full debugging and never crashes on empty."""
    print("🔨 Building FAISS index from knowledge/ and ALL subfolders...")
=======
    """Build or rebuild the full FAISS index from knowledge/ (including ALL nested folders)."""
    print("Building FAISS index from knowledge/ and all subfolders...")
>>>>>>> Stashed changes

    knowledge_dir = Path("knowledge").resolve()
    print(f"→ Scanning absolute path: {knowledge_dir}")
    print(f"   Folder exists: {knowledge_dir.exists()}")
    print(f"   Is directory: {knowledge_dir.is_dir()}")

<<<<<<< Updated upstream
    if not knowledge_dir.exists():
        print("❌ 'knowledge' folder not found in current directory!")
        print("   Run: mkdir -p knowledge")
        return None

    # Debug: count files manually (super visible)
    pdf_files = list(knowledge_dir.rglob("*.pdf")) + list(knowledge_dir.rglob("*.PDF"))
    md_files  = list(knowledge_dir.rglob("*.md"))  + list(knowledge_dir.rglob("*.MD"))
    txt_files = list(knowledge_dir.rglob("*.txt")) + list(knowledge_dir.rglob("*.TXT"))
    csv_files = list(knowledge_dir.rglob("*.csv")) + list(knowledge_dir.rglob("*.CSV"))

    print(f"Found {len(pdf_files)} PDF(s), {len(md_files)} MD(s), {len(txt_files)} TXT(s), {len(csv_files)} CSV(s)")

    if len(pdf_files) + len(md_files) + len(txt_files) + len(csv_files) == 0:
        print("⚠️  NO supported files found!")
        print("   Drop some .pdf / .md / .txt / .csv into knowledge/ or any subfolder")
        print("   Example: knowledge/meeting_notes/lsic_prep.md")
        return None

    # Now load with recursive=True (this is what you fixed before launch)
    pdf_docs = DirectoryLoader(
        str(knowledge_dir), glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True, use_multithreading=True, recursive=True
=======
    # PDFs - recursive
    pdf_docs = DirectoryLoader(
        knowledge, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True, 
        use_multithreading=True,
        recursive=True          # ← This was the missing piece
>>>>>>> Stashed changes
    ).load()

    md_docs = DirectoryLoader(
<<<<<<< Updated upstream
        str(knowledge_dir), glob="**/*.md", loader_cls=TextLoader,
        show_progress=True, use_multithreading=True, recursive=True
=======
        knowledge, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        show_progress=True, 
        use_multithreading=True,
        recursive=True
>>>>>>> Stashed changes
    ).load()

    txt_docs = DirectoryLoader(
<<<<<<< Updated upstream
        str(knowledge_dir), glob="**/*.txt", loader_cls=TextLoader,
        show_progress=True, use_multithreading=True, recursive=True
=======
        knowledge, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        show_progress=True, 
        use_multithreading=True,
        recursive=True
>>>>>>> Stashed changes
    ).load()

    csv_docs = DirectoryLoader(
<<<<<<< Updated upstream
        str(knowledge_dir), glob="**/*.csv", loader_cls=CSVLoader,
        show_progress=True, use_multithreading=True, recursive=True
    ).load()

    all_docs = pdf_docs + md_docs + txt_docs + csv_docs
    print(f"✅ Successfully loaded {len(all_docs)} documents (including nested folders)")
=======
        knowledge, 
        glob="**/*.csv", 
        loader_cls=CSVLoader,
        show_progress=True, 
        use_multithreading=True,
        recursive=True
    ).load()

    all_docs = pdf_docs + md_docs + txt_docs + csv_docs
    print(f"Loaded {len(all_docs)} documents from knowledge/ (including nested folders)")
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream

if __name__ == "__main__":
    build_index()
=======
>>>>>>> Stashed changes
