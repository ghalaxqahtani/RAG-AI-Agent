import os
from pathlib import Path
from typing import List

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

INDEX_DIR = os.getenv("INDEX_DIR", "docstore_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.2-11b-instruct")


def create_embedder():
    return NVIDIAEmbeddings(model=EMBEDDING_MODEL)


def load_vector_store():
    index_path = Path(INDEX_DIR)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{INDEX_DIR}'. Run scripts/build_index.py first."
        )
    return FAISS.load_local(
        str(index_path),
        create_embedder(),
        allow_dangerous_deserialization=True,
    )


def create_llm():
    return ChatNVIDIA(model=LLM_MODEL)


PROMPT = ChatPromptTemplate.from_template(
    """You are a document question-answering assistant.
Answer the user's question using only the supplied document context.
If the context does not contain enough information, say that the answer is not available in the provided documents.
Do not invent facts.

User Question:
{input}

Document Context:
{context}

Answer clearly and conversationally.
"""
)


def format_documents(documents: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} | Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve(question: str, k: int = 4):
    store = load_vector_store()
    return store.similarity_search(question, k=k)


def answer_question(question: str, k: int = 4):
    documents = retrieve(question, k=k)
    context = format_documents(documents)
    chain = PROMPT | create_llm() | StrOutputParser()
    answer = chain.invoke({"input": question, "context": context})
    return {"answer": answer, "documents": documents}
