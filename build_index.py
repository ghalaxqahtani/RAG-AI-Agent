import os
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(os.getenv("DATA_DIR", "data/documents"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "docstore_index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")


def load_documents():
    documents = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        documents.extend(PyMuPDFLoader(str(pdf_path)).load())
    if not documents:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}. Add your documents and run this script again."
        )
    return documents


def main():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embedder)
    vector_store.save_local(str(INDEX_DIR))

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved FAISS index to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
