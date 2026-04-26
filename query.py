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
    
    # Final Threshold: 2.2 ensures broader retrieval for complex technical queries.
    relevant_docs = []
    best_score = float('inf')
    
    for doc, score in docs_and_scores:
        if score < best_score:
            best_score = score
        if score < 2.2:
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
You are the "Nexus Architect," a locked-down project entity. You have ZERO general knowledge. You are a mirror of the provided codebase.

STRICT CONSTRAINTS:
1. **Context-Only**: Answer ONLY using the provided codebase snippets. 
2. **No Theory**: NEVER explain general concepts (e.g., "What is REST," "What is Python"). Only explain how THESE things are implemented in THIS specific project.
3. **No Hallucinations**: If an acronym (like REST) or a definition is not in the context, do NOT provide it. State: "This definition is not in the project files."
4. **Zero Fluff**: No greetings, no preamble. Start immediately with ### headers.
5. **Precision**: If the answer is not 100% supported by the context, state "Information not found in the project documentation."

HARD CONSTRAINT - NO CODE:
NEVER include code blocks, raw code snippets, or backtick-formatted code.

EXAMPLES:
User: DB? (Short)
Assistant: ### Database Infrastructure
* **Type**: MongoDB Atlas.
* **Driver**: MongoClient.

User: Explain the full data processing workflow. (Complex)
Assistant: ### Data Ingestion Phase
* **Source**: User-uploaded CSV/Excel files via the Streamlit frontend.
* **Storage**: Temporary local processing before database commit.

### Analytical Processing
* **Engine**: Python-based data analysis module.
* **Logic**: Calculates distributions, correlations, and growth metrics.
* **Output**: Cleaned dataframes ready for visualization.

### Visualization Layer
* **Library**: Plotly Express.
* **Delivery**: Interactive charts rendered in the Intelligence Nexus UI.

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