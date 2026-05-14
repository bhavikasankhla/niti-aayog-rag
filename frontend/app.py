"""
app.py — Streamlit Frontend

Two sections:
  Sidebar : Upload PDF + show processing status
  Main    : Tabs for Insights and Q&A Chat
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Niti Aayog RAG Assistant",
    page_icon="📊",
    layout="wide",
)

# ── Session State ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "insights" not in st.session_state:
    st.session_state.insights = []

if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False


# ── Helper: check API status ──────────────────────────────
def check_status():
    try:
        r = requests.get(f"{API_URL}/status", timeout=5)
        return r.json().get("indexed", False)
    except Exception:
        return False


# ── Sidebar: Upload ───────────────────────────────────────
with st.sidebar:
    st.title("📊 Niti Aayog RAG")
    st.caption("AI-powered Annual Report Assistant")
    st.divider()

    st.subheader("1. Upload Report")
    uploaded_file = st.file_uploader(
        "Choose a Niti Aayog Annual Report PDF",
        type=["pdf"],
        help="Upload the PDF to index it for Q&A"
    )

    if uploaded_file:
        if st.button("⚙️ Process Document", use_container_width=True, type="primary"):
            with st.spinner("Extracting text, chunking, embedding... (may take 1-2 min)"):
                try:
                    r = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=300,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ Indexed {data['chunks_indexed']} chunks!")
                        st.session_state.doc_indexed = True
                        st.session_state.insights = []  # reset so insights re-fetch
                        st.session_state.chat_history = []
                    else:
                        st.error(f"Error: {r.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is `uvicorn backend.main:app` running?")

    st.divider()

    # Show current status
    st.subheader("Status")
    if check_status():
        st.success("✅ Document indexed & ready")
    else:
        st.warning("⚠️ No document indexed yet")

    st.divider()
    st.caption("Built with LangChain + ChromaDB + Gemini")


# ── Main Area ─────────────────────────────────────────────
st.title("📊 Niti Aayog Annual Report Assistant")
st.caption("Upload a report from the sidebar, then explore insights or ask questions.")

tab_insights, tab_qa = st.tabs(["💡 Key Insights", "💬 Ask Questions"])


# ── Tab 1: Insights ───────────────────────────────────────
with tab_insights:
    st.subheader("Auto-Generated Key Insights")
    st.caption("The AI extracts the most important points from the report automatically.")

    if st.button("🔍 Generate Insights", use_container_width=False):
        if not check_status():
            st.warning("Please upload and process a PDF first.")
        else:
            with st.spinner("Analyzing document and extracting insights..."):
                try:
                    r = requests.get(f"{API_URL}/insights", timeout=300)
                    if r.status_code == 200:
                        st.session_state.insights = r.json().get("insights", [])
                    else:
                        st.error(r.json().get("detail", "Error fetching insights"))
                except Exception as e:
                    st.error(f"Connection error: {e}")

    if st.session_state.insights:
        for i, insight in enumerate(st.session_state.insights, 1):
            with st.container():
                st.markdown(f"""
                <div style="
                    background: #f0f4ff;
                    border-left: 4px solid #4f6ef7;
                    padding: 12px 16px;
                    border-radius: 4px;
                    margin: 8px 0;
                    color: #1a1a2e;
                ">
                    <strong>#{i}</strong> &nbsp; {insight}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Insights' after uploading a document.")


# ── Tab 2: Q&A Chat ───────────────────────────────────────
with tab_qa:
    st.subheader("Ask Questions About the Report")
    st.caption("Questions are answered using only the document — no hallucinations.")

    # Display chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sources"):
                with st.expander("📄 Source Chunks (retrieved from document)"):
                    for j, src in enumerate(turn["sources"], 1):
                        st.caption(f"**Chunk {j}:**")
                        st.text(src)

    # Question input
    question = st.chat_input("Ask something about the Niti Aayog report...")

    if question:
        if not check_status():
            st.warning("Please upload and process a PDF first.")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant sections and generating answer..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/ask",
                            json={"question": question},
                            timeout=60,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            st.write(data["answer"])

                            if data.get("sources"):
                                with st.expander("📄 Source Chunks (retrieved from document)"):
                                    for j, src in enumerate(data["sources"], 1):
                                        st.caption(f"**Chunk {j}:**")
                                        st.text(src)

                            # Save to history
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": data["answer"],
                                "sources": data.get("sources", []),
                            })
                        else:
                            st.error(r.json().get("detail", "Error getting answer"))

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend. Is it running?")

    # Sample questions
    if not st.session_state.chat_history:
        st.divider()
        st.caption("💡 Try these sample questions:")
        sample_qs = [
            "What are the major economic targets mentioned?",
            "Summarize the agriculture sector highlights",
            "What are the key infrastructure projects?",
            "Which government schemes are mentioned?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(sample_qs):
            with cols[i % 2]:
                st.code(q, language=None)