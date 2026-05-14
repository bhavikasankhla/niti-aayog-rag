"""
rag.py — RAG Pipeline
Embeddings : HuggingFace all-MiniLM-L6-v2 (local)
LLM        : Ollama llama3.2 (local)
Vector DB  : ChromaDB (local)
No API keys. No quotas. Runs fully offline.
"""

import fitz
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    EMBEDDING_MODEL,
    LLM_MODEL,
)


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        full_text += f"\n[Page {page_num + 1}]\n{page.get_text()}"
    doc.close()
    return full_text


def split_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.create_documents([text])


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


def index_document(pdf_path: str) -> int:
    print(f"[RAG] Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    print("[RAG] Splitting into chunks...")
    chunks = split_text(text)
    print(f"[RAG] Created {len(chunks)} chunks")
    print("[RAG] Embedding with HuggingFace (local)...")
    embeddings = get_embeddings()
    try:
        existing = Chroma(collection_name=CHROMA_COLLECTION, embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
        existing.delete_collection()
    except Exception:
        pass
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"[RAG] Indexed {len(chunks)} chunks successfully.")
    return len(chunks)


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert analyst of documents.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
)


def answer_question(question: str) -> dict:
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_RESULTS})
    relevant_docs = retriever.invoke(question)
    if not relevant_docs:
        return {"answer": "No document indexed yet. Please upload a PDF first.", "sources": []}
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    prompt = QA_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    sources = [doc.page_content[:300] + "..." for doc in relevant_docs]
    return {"answer": response.content, "sources": sources}


INSIGHTS_PROMPT = """You are an expert analyst. Based on these document excerpts, list the 8 most important insights.
Focus on key targets, schemes, statistics, and priorities.
Format as a numbered list. Each point 1-2 sentences.

Document excerpts:
{context}

Top 8 Key Insights:"""


def generate_insights() -> List[str]:
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    broad_queries = ["key achievements and targets", "major schemes and programs", "statistics and numbers", "strategic priorities"]
    all_docs = []
    for query in broad_queries:
        docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(query)
        all_docs.extend(docs)
    seen = set()
    unique_docs = [doc for doc in all_docs if doc.page_content not in seen and not seen.add(doc.page_content)]
    context = "\n\n---\n\n".join([doc.page_content for doc in unique_docs[:6]])
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    response = llm.invoke(INSIGHTS_PROMPT.format(context=context))
    lines = response.content.strip().split("\n")
    insights = [l.strip() for l in lines if l.strip() and l.strip()[0].isdigit()]
    return insights if insights else [response.content]


def is_document_indexed() -> bool:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        for col in client.list_collections():
            if col.name == CHROMA_COLLECTION:
                return col.count() > 0
        return False
    except Exception:
        return False