# Project Architecture: Intelligence Nexus & DataForge AI

## Backend Infrastructure
- **API Framework**: FastAPI (High-performance Python API).
- **Architecture Style**: **REST API** (Representational State Transfer).
- **Database**: MongoDB Atlas (Cloud NoSQL database).
- **Environment**: Python 3.10+.

## RAG Intelligence
- **System Type**: Retrieval-Augmented Generation (RAG).
- **Vector Database**: FAISS (Facebook AI Similarity Search).
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`).
- **LLM Engine**: Llama 3.1 8B (via Groq LPU).

## Frontend Implementation
- **Framework**: Streamlit.
- **Design System**: Electric Red / Beige Neon theme.
- **Components**: Custom CSS chat bubbles with human avatar integration.

## Communication Protocol
- The system uses **REST API** endpoints (GET/POST) to communicate between the Streamlit frontend and the FastAPI backend.
