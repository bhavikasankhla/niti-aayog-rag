"""
main.py — FastAPI Backend

Endpoints:
  POST /upload      → Upload PDF, trigger indexing
  GET  /insights    → Get auto-generated insights
  POST /ask         → Ask a question, get RAG answer
  GET  /health      → Health check
  GET  /status      → Check if document is indexed
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import index_document, answer_question, generate_insights, is_document_indexed
from backend.config import DATA_DIR

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Niti Aayog RAG Assistant",
    description="Ask questions about Niti Aayog Annual Reports using RAG",
    version="1.0.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ── Request/Response Models ────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int

class InsightsResponse(BaseModel):
    insights: list[str]

class StatusResponse(BaseModel):
    indexed: bool
    message: str


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check."""
    return {"status": "ok", "service": "Niti Aayog RAG Assistant"}


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Check if a document has been indexed and is ready for queries."""
    indexed = is_document_indexed()
    return StatusResponse(
        indexed=indexed,
        message="Document is ready." if indexed else "No document indexed yet. Please upload a PDF."
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a Niti Aayog Annual Report PDF.
    Triggers full ingestion pipeline: extract → chunk → embed → store.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to disk
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion pipeline
    try:
        chunks_count = index_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        filename=file.filename,
        chunks_indexed=chunks_count,
    )


@app.get("/insights", response_model=InsightsResponse)
def get_insights():
    """
    Auto-generate key insights from the indexed document.
    Must call /upload first.
    """
    if not is_document_indexed():
        raise HTTPException(
            status_code=400,
            detail="No document indexed. Upload a PDF first via /upload."
        )

    try:
        insights = generate_insights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

    return InsightsResponse(insights=insights)


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about the indexed document.
    Returns an answer grounded in the document + source chunk previews.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not is_document_indexed():
        raise HTTPException(
            status_code=400,
            detail="No document indexed. Upload a PDF first via /upload."
        )

    try:
        result = answer_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QuestionResponse(
        answer=result["answer"],
        sources=result["sources"],
    )
