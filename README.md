# 💠 Intelligence Nexus | High-Precision RAG Engine

**Intelligence Nexus** is a premium, zero-hallucination RAG (Retrieval-Augmented Generation) chatbot designed for surgical codebase analysis. Built with a "Noise-Free" philosophy, it provides direct, data-driven answers from your project documentation without conversational fluff or hallucinations.

---

## 🚀 Key Capabilities

### 🔍 **Surgical Retrieval**
- **Semantic Precision**: Powered by **FAISS** and **HuggingFace Embeddings**, it understands the meaning behind your code, not just keywords.
- **Deep Context**: Utilizes a 1000-character chunking strategy to maintain logical technical context.
- **Source Transparency**: Every answer includes high-contrast "Reference Tags" pointing to the exact source files.

### 🧠 **Neural Architecture**
- **LLM**: Powered by **Llama 3.1 8B** via the high-speed **Groq LPU** inference engine.
- **Orchestration**: Built on **LangChain** for robust document management and retrieval chains.
- **Zero-Code Policy**: Technical logic is explained in natural language for maximum clarity (Hard Constraint).

### 📱 **Premium UI/UX**
- **Electric Red Aesthetic**: A bold, high-contrast dark mode design with sophisticated beige accents.
- **Dynamic Scaling**: AI response depth automatically adjusts based on the complexity of your question.
- **Human Identity**: Integrated professional human avatar for a personalized user presence.

---

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **AI Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Meta AI)
- **Inference Engine**: [Groq](https://groq.com/)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Language Model**: Llama 3.1 8B

---

## 🏁 Quick Start

### 1. Installation
```bash
git clone https://github.com/Karthi02A/RAG-System.git
cd RAG-System
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root:
```env
GROQ_API_KEY=your_api_key_here
```

### 3. Build the Brain (Ingest)
```bash
python ingest.py
```

### 4. Launch the Nexus
```bash
streamlit run chat_ui.py
```

---

## 🗺️ Project Structure

- `chat_ui.py`: The "Face" (Premium Streamlit Interface).
- `query.py`: The "Thinker" (RAG logic & Groq integration).
- `ingest.py`: The "Librarian" (Vector DB creation).
- `vector_store/`: The "Brain" (FAISS Index).
- `project_data/`: The "Knowledge Base" (Analyzed files).
- `assets/`: The "Branding" (Avatars & Visuals).

---

## 🌐 Deployment
Optimized for **Streamlit Cloud**.  
*Zero-Downtime. Zero-Hallucination. 100% Intelligence.*

Built by **Antigravity** ⚒️✨
