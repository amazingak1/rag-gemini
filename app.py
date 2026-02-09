import streamlit as st
import os
import tempfile

from vectorstore import build_vectorstore
from rag import run_rag
from config import HIGHLIGHT_BG_COLOR, HIGHLIGHT_BORDER_COLOR

# ---------------- UI ----------------
st.set_page_config(page_title="PDF RAG with Highlighting")
st.title("📘 Chat with PDF (RAG + Highlighted Sources)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

vectorstore = None

# ---------------- PDF PROCESSING ----------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    try:
        vectorstore = build_vectorstore(pdf_path)
        st.success("✅ PDF indexed successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.stop()

# ---------------- QUERY ----------------
query = st.text_input(
    "Ask a question from the PDF",
    placeholder="e.g., What is the main topic?"
)

if query and vectorstore:
    with st.spinner("Searching for relevant information..."):
        answer, docs = run_rag(query, vectorstore)

    # -------- ANSWER --------
    st.subheader("💡 Answer")
    st.write(answer)

    # -------- SOURCES --------
    st.subheader("📄 Sources (Retrieved Chunks)")
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "N/A")
        source = os.path.basename(doc.metadata.get("source", "PDF"))

        with st.expander(f"📍 Source {i}: {source} (Page {page})"):
            st.markdown(
                f"""
                <div style="
                    background-color:{HIGHLIGHT_BG_COLOR};
                    padding:12px;
                    border-left:6px solid {HIGHLIGHT_BORDER_COLOR};
                ">
                {doc.page_content}
                </div>
                """,
                unsafe_allow_html=True
            )

elif query:
    st.warning("Please upload a PDF first.")
