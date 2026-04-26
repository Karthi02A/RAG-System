from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load DB
db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# Test query
query = "authentication logic"

docs = db.similarity_search(query, k=3)

print("\nTop results:\n")
for i, d in enumerate(docs):
    print(f"--- Result {i+1} ---")
    print(d.page_content[:300])
    print()