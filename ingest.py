import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Map extensions to LangChain Language enums safely
def get_language_enum(ext):
    mapping = {
        ".py": "PYTHON",
        ".js": "JS",
        ".jsx": "JS",
        ".ts": "TS",
        ".tsx": "TS",
        ".html": "HTML",
        ".css": "CSS",
        ".sql": "SQL",
        ".md": "MARKDOWN",
        ".txt": None,
        "Dockerfile": None,
        ".yml": None,
        ".yaml": None,
        ".json": None,
    }
    lang_name = mapping.get(ext)
    if lang_name and hasattr(Language, lang_name):
        return getattr(Language, lang_name)
    return None

IGNORE_DIRS = {".git", "venv", "__pycache__", "node_modules", "vector_store"}

def load_project(folder):
    docs = []
    for root, dirs, files in os.walk(folder):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            # Special case for Dockerfile which has no extension
            filename_key = file if not ext else ext
            
            if get_language_enum(filename_key) or filename_key in [".txt", ".css", ".md", ".sql", ".py", ".js", ".ts", ".html", "Dockerfile", ".yml", ".yaml", ".json"]:
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, folder)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        docs.append({
                            "content": f.read(),
                            "path": rel_path,
                            "ext": ext
                        })
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return docs

# 1. Load project
documents = load_project("project_data")
print(f"Total files found in 'project_data': {len(documents)}")

# 2. Chunking with Language Awareness
chunks = []
for doc in documents:
    filename_key = doc["path"].split(os.sep)[-1]
    ext = doc["ext"]
    lang = get_language_enum(ext if ext else filename_key)
    splitter = None
    
    if lang:
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=1000,
                chunk_overlap=200
            )
        except Exception as e:
            print(f"Warning: Could not create language splitter for {lang}: {e}")
            splitter = None
            
    if not splitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    parts = splitter.split_text(doc["content"])
    for part in parts:
        chunks.append({
            "text": part,
            "source": doc["path"]
        })

print(f"Total chunks generated: {len(chunks)}")

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts = [c["text"] for c in chunks]
metas = [{"source": c["source"]} for c in chunks]

# 4. Vector DB
if chunks:
    db = FAISS.from_texts(texts, embeddings, metadatas=metas)
    db.save_local("vector_store")
    print("[SUCCESS] Vector DB updated successfully in 'vector_store'")
else:
    print("[ERROR] No chunks generated. Vector DB not updated.")