from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import answer_question, retrieve

app = FastAPI(
    title="RAG Document Chatbot API",
    version="1.0.0",
    description="A retrieval-augmented document question-answering API built with LangChain, FAISS, FastAPI, and NVIDIA NIM-compatible models.",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retriever")
def retriever_endpoint(request: QuestionRequest):
    try:
        docs = retrieve(request.question, request.k)
        return {
            "documents": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask_endpoint(request: QuestionRequest):
    try:
        result = answer_question(request.question, request.k)
        return {
            "answer": result["answer"],
            "sources": [d.metadata for d in result["documents"]],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
