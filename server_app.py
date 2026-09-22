
from fastapi import FastAPI

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS


# ============================================================
# Models
# ============================================================

embedder = NVIDIAEmbeddings(
    model="course/embedding",
    base_url="http://llm_client:9000/v1",
)

instruct_llm = ChatNVIDIA(
    model="meta/llama-3.2-11b-vision-instruct",
    model_kwargs={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using LangChain's Runnable interfaces",
)


# ============================================================
# Basic Chat
# ============================================================

add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)


# ============================================================
# Load Document Store
# ============================================================

docstore = FAISS.load_local(
    "docstore_index",
    embedder,
    allow_dangerous_deserialization=True,
)


# ============================================================
# Retriever Endpoint
# ============================================================

def retrieve_documents(x):
    return docstore.similarity_search(x, k=4)


retriever = RunnableLambda(retrieve_documents).with_types(
    input_type=str
)

add_routes(
    app,
    retriever,
    path="/retriever",
)


# ============================================================
# Generator Endpoint
# ============================================================

generator_prompt = ChatPromptTemplate.from_template(
    """You are a document chatbot.
Answer the user's question using only the provided document context.

User Question:
{input}

Document Context:
{context}

Answer conversationally and only use information supported by the context.
"""
)

generator_chain = (
    generator_prompt
    | instruct_llm
    | StrOutputParser()
)

add_routes(
    app,
    generator_chain,
    path="/generator",
)


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
