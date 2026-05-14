# 🔍 RAG-based PDF Document Analyzer

An end-to-end **Retrieval-Augmented Generation (RAG)** system that lets you upload any PDF and:
- 💡 **Auto-extract key insights** from the document
- 💬 **Ask questions** and get accurate answers grounded in the document
- 🚫 **No hallucinations** — answers only from your document

> 100% local — no API keys, no costs, no internet required after setup.

---

## 🎥 Demo

| Feature | Description |
|---|---|
| Upload any PDF | Government reports, research papers, legal docs, manuals |
| Auto Insights | AI extracts top 8 key points automatically |
| Q&A Chat | Ask anything, get answers with source references |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Text Chunking (LangChain RecursiveCharacterTextSplitter)
    │
    ▼
Embeddings (HuggingFace all-MiniLM-L6-v2 — runs locally)
    │
    ▼
Vector Store (ChromaDB — persisted locally)
    │
    ▼
Query → Retrieve Top-K Chunks → LLM (Ollama llama3.2 — runs locally) → Answer
```

**Tech Stack:**

| Component | Technology | Why |
|---|---|---|
| RAG Framework | LangChain | Industry standard, clean abstractions |
| Vector DB | ChromaDB | Local, no infra, easy to swap to Pinecone in prod |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free, local, no API key, fast |
| LLM | Ollama llama3.2 | Runs locally, zero cost, no quota limits |
| Backend | FastAPI | Async, auto-generated API docs |
| Frontend | Streamlit | Clean UI, fast to iterate |

---

## 📁 Project Structure

```
pdf-analyzer/
├── backend/
│   ├── main.py       # FastAPI — 4 REST endpoints
│   ├── rag.py        # Core RAG pipeline
│   └── config.py     # Settings & model config
├── frontend/
│   └── app.py        # Streamlit UI
├── data/             # PDFs stored here (gitignored)
├── chroma_db/        # Vector store (gitignored)
├── scripts/
│   └── test_rag.py   # CLI tester
└── requirements.txt
```

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Clone the repo
```bash
git clone https://github.com/bhavikasankhla/pdf-analyzer.git
cd pdf-analyzer
```

### 2. Pull the LLM model
```bash
ollama pull llama3.2
```

### 3. Create virtual environment
```bash
python -m venv .venv

# Mac/Linux:
source .venv/bin/activate

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
pip install sentence-transformers langchain-ollama langchain-huggingface
```

### 5. Run backend (Terminal 1)
```bash
uvicorn backend.main:app --reload --port 8000
```

### 6. Run frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501**

---

## 🔄 How to Use

1. **Upload** any PDF from the sidebar
2. Click **"Process Document"** — chunks and indexes it (~1-2 min)
3. Click **"Generate Insights"** — AI extracts key points (~2-3 min for large docs)
4. Switch to **"Ask Questions"** tab — chat with your document

---

## 🧠 RAG Pipeline (Interview Ready)

| Step | What Happens |
|---|---|
| **Ingestion** | PDF → extract text → split into 1000-char chunks with 200-char overlap |
| **Embedding** | Each chunk → HuggingFace model → 384-dim vector stored in ChromaDB |
| **Retrieval** | User query → embed → cosine similarity → top 5 most relevant chunks |
| **Generation** | Retrieved chunks + question → Ollama LLM → grounded answer |

**Key design decisions:**
- **Chunk overlap** prevents context loss at chunk boundaries
- **Local embeddings** eliminate API costs and latency
- **Prompt engineering** constrains LLM to only use retrieved context — no hallucinations
- **ChromaDB persistence** means re-indexing not needed between sessions

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload & index a PDF |
| `GET` | `/insights` | Auto-generate key insights |
| `POST` | `/ask` | Ask a question |
| `GET` | `/status` | Check if document is indexed |
| `GET` | `/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

---

## ⚙️ Configuration

Edit `backend/config.py` to customize:

```python
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between chunks
TOP_K_RESULTS = 5        # chunks retrieved per query
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace model
LLM_MODEL = "llama3.2"                  # Ollama model
```

---

## 📈 Scaling to Production

| Current | Production Swap |
|---|---|
| ChromaDB (local) | Pinecone / Weaviate |
| Ollama llama3.2 | OpenAI GPT-4 / Claude |
| Single PDF | Multi-document with metadata filters |
| No auth | JWT authentication |

---

## 🛠️ Built With

`Python` `LangChain` `ChromaDB` `Ollama` `HuggingFace` `FastAPI` `Streamlit` `PyMuPDF`