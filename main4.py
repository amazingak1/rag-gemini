import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# NEW Gemini SDK
from google import genai

# ---------------- ENV ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------- UI ----------------
st.set_page_config(page_title="PDF RAG with Highlighting", page_icon="📘")
st.title("📘 Chat with PDF (RAG + Highlighted Sources)4")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


# ---------------- EMBEDDINGS (ONCE) ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# ---------------- VECTORSTORE ----------------
@st.cache_resource
def build_vectorstore(pdf_bytes, _embeddings):
    """
    Build vectorstore from PDF bytes.
    Using pdf_bytes ensures caching works correctly for different PDFs.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    try:
        # Load and process PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        # Build vectorstore
        vectorstore = FAISS.from_documents(chunks, _embeddings)

        return vectorstore

    finally:
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


# Initialize vectorstore
vectorstore = None

if uploaded_file:
    # Read PDF bytes
    pdf_bytes = uploaded_file.read()

    with st.spinner("Processing PDF... This may take a moment."):
        try:
            vectorstore = build_vectorstore(pdf_bytes, embeddings)
            st.success("✅ PDF indexed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()

# ---------------- QUERY ----------------
query = st.text_input("Ask a question from the PDF", placeholder="e.g., What is the main topic?")

if query and vectorstore:
    with st.spinner("Searching for relevant information..."):
        try:
            # Retrieve relevant documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)

            # Build context from retrieved documents
            context = "\n\n".join(doc.page_content for doc in docs)

            # Create prompt
            prompt = f"""
Answer ONLY using the context below.
If the answer is not present in the context, say "I don't know based on the provided document."
Try to answer in short and concise manner.

Context:
{context}

Question:
{query}
"""

            # Generate response using Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )

            # -------- ANSWER --------
            st.subheader("💡 Answer")
            st.write(response.text)

            # -------- SOURCES --------
            st.subheader("📄 Sources (Retrieved Chunks)")
            for i, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", "N/A")

                # Adjust page number (PyPDFLoader uses 0-indexing)
                if isinstance(page, int):
                    page = page + 1

                source = os.path.basename(doc.metadata.get("source", "PDF"))

                with st.expander(f"📍 Source {i}: {source} (Page {page})"):
                    st.markdown(
                        f"""
                        <div style="background-color:#fff3cd;
                                    padding:12px;
                                    border-radius:5px;
                                    border-left:6px solid #ffcc00;">
                        {doc.page_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error generating response: {e}")

elif query:
    st.warning("⚠️ Please upload a PDF first.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with Streamlit, LangChain, FAISS, and Google Gemini
    </div>
    """,
    unsafe_allow_html=True
)

# Run with: streamlit run main4.py