# RAG Document Chatbot

A document question-answering chatbot built using Retrieval-Augmented Generation (RAG).

The project allows users to add their own PDF documents, build a searchable vector index, retrieve relevant document content, and generate answers based on the retrieved context.

![RAG Document Chatbot](assets/rag-agent-demo.png)

---

## Project Overview

This project implements a practical RAG pipeline for document-based question answering.

Instead of relying only on the language model's internal knowledge, the system retrieves relevant information from the user's documents and provides that context to the language model before generating an answer.

### RAG Workflow

```text
PDF Documents
      ↓
Document Processing
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Semantic Retrieval
      ↓
Relevant Document Context
      ↓
LLM
      ↓
Generated Answer
