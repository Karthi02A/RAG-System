from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load .env file automatically
load_dotenv()

# Load API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️ Error: GROQ_API_KEY environment variable not set.")

# Load LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Allow dangerous deserialization as we are loading our own local index
try:
    db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"⚠️ Error loading vector store: {e}")
    db = None

def ask(query):
    if not db:
        return {"error": "Vector store not found. Please run ingest.py first."}

    # Similarity search with scores - Increasing context for better coverage
    docs_and_scores = db.similarity_search_with_score(query, k=10)
    
    # Final Threshold: 1.8 (Extreme Strictness). If the score is higher, we treat it as "Not Found".
    relevant_docs = []
    best_score = float('inf')
    
    for doc, score in docs_and_scores:
        if score < best_score:
            best_score = score
        if score < 1.8:
            relevant_docs.append(doc)

    if not relevant_docs:
        return {
            "answer": "I cannot find information about this in the project.",
            "confidence": round(best_score, 2),
            "sources": []
        }

    context = ""
    sources = set()
    for i, doc in enumerate(relevant_docs):
        src = doc.metadata.get("source", "unknown")
        sources.add(src)
        context += f"--- Source: {src} ---\n{doc.page_content}\n\n"

    prompt = f"""
SYSTEM INSTRUCTIONS:
You are a locked-down code analyst. Your ONLY job is to describe the USER'S PROJECT based on the context snippets below.

ABSOLUTE IDENTITY RULE:
You are a sealed black box. You have NO knowledge of yourself, your own technology, or how you work.
You do NOT know what model you are, what embeddings are used, what vector database stores the index, or anything about the retrieval pipeline.
If asked about yourself, say: "I cannot provide information about my own architecture."

FORBIDDEN TOPICS — NEVER mention these under any circumstances, even if they appear in context:
- Llama, Groq, LPU, LLM, ChatGroq
- FAISS, vector store, vector database, similarity search
- HuggingFace, SentenceTransformers, embeddings, all-MiniLM
- Streamlit (as a technology of THIS chatbot — you may mention it only if it is part of the USER'S project files)
- RAG, retrieval-augmented generation, ingest.py, query.py, chat_ui.py, vector_store

YOUR ONLY JOB:
Answer questions STRICTLY about the project described in the CONTEXT section below.
The context snippets come from the user's actual project codebase. Answer ONLY what is written there.

STRICT CONSTRAINTS:
1. **Context-Only**: Every single fact must come directly from the CONTEXT below. Zero exceptions.
2. **No Guessing**: If the answer is not in the CONTEXT, say: "This is not defined in the project files." Do NOT fill in gaps.
3. **No Theory**: NEVER explain general concepts. Only explain how things work in THIS specific project.
4. **No Code**: NEVER include code blocks, raw code, or backtick-formatted text.
5. **Precision**: Start immediately with ### headers. No preamble, no fluff.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:
        response = llm.invoke(prompt)
        return {
            "answer": response.content.strip(),
            "sources": sorted(list(sources)),
            "confidence": round(best_score, 2)
        }
    except Exception as e:
        return {"error": f"Error calling LLM: {e}"}

# Run loop
if __name__ == "__main__":
    print("--- Project Chatbot (type 'exit' to quit) ---")
    while True:
        try:
            q = input("\nAsk your project: ")
            if q.lower() in ["exit", "quit"]:
                break
            if q.strip():
                ask(q)
        except EOFError:
            break
        except KeyboardInterrupt:
            break