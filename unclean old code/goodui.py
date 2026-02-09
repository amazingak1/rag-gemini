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
    st.error("🔑 GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="PDF RAG with Highlighting",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .source-chip {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin-right: 8px;
        font-weight: 500;
    }
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1976d2;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        chunk_size = st.slider("Chunk Size", 400, 1200, 800, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 300, 150, 50)
        num_results = st.slider("Number of Sources", 1, 5, 3)

        answer_style = st.selectbox(
            "Answer Style",
            ["Concise", "Detailed", "Bullet Points"]
        )

    st.markdown("---")

    # Info section
    st.markdown("### 📊 About")
    st.info("""
    This RAG (Retrieval-Augmented Generation) system:
    - Indexes your PDF documents
    - Retrieves relevant chunks
    - Generates accurate answers using Gemini
    """)

    st.markdown("### 💡 Tips")
    st.markdown("""
    - Upload clear, text-based PDFs
    - Ask specific questions
    - Check sources for verification
    """)

# ---------------- MAIN CONTENT ----------------
# Header
st.markdown('<p class="main-header">📘 Chat with Your PDF</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF and ask questions using AI-powered retrieval</p>',
            unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📎 Upload PDF Document",
        type=["pdf"],
        help="Upload a PDF file to start asking questions"
    )

with col2:
    if uploaded_file:
        st.success("✅ File uploaded!")
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.metric("File Size", f"{file_size:.1f} KB")


# ---------------- EMBEDDINGS (ONCE) ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = get_embeddings()


# ---------------- VECTORSTORE ----------------
@st.cache_resource
def build_vectorstore(pdf_path, _chunk_size, _chunk_overlap):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    return FAISS.from_documents(chunks, embeddings), len(documents), len(chunks)


vectorstore = None
num_pages = 0
num_chunks = 0

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    try:
        with st.spinner("🔄 Processing PDF... This may take a moment"):
            vectorstore, num_pages, num_chunks = build_vectorstore(pdf_path, chunk_size, chunk_overlap)

        # Success metrics in cards
        st.markdown("### 📈 Document Statistics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("📄 Pages", num_pages)
        with metric_col2:
            st.metric("🧩 Chunks", num_chunks)
        with metric_col3:
            st.metric("🔍 Indexed", "Yes" if vectorstore else "No")

    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        st.stop()

st.markdown("---")

# ---------------- QUERY ----------------
# Chat-like interface
st.markdown("### 💬 Ask Your Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (Page {source['page']})")
                    st.markdown(f"> {source['content'][:200]}...")

# Query input
query = st.chat_input("Ask a question about the PDF..." if vectorstore else "Please upload a PDF first")

if query and vectorstore:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": num_results})
            docs = retriever.invoke(query)

            context = "\n\n".join(doc.page_content for doc in docs)

            # Customize prompt based on style
            style_instructions = {
                "Concise": "Answer ONLY using the context below in 2-3 sentences.",
                "Detailed": "Answer using the context below with detailed explanation.",
                "Bullet Points": "Answer using the context below in bullet points."
            }

            prompt = f"""
{style_instructions.get(answer_style, style_instructions["Concise"])}
If the answer is not present, say "I don't know based on the provided document."

Context:
{context}

Question:
{query}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            # Display answer
            st.markdown(f'<div class="answer-box">{response.text}</div>', unsafe_allow_html=True)

            # Sources in expander
            with st.expander(f"📚 View {len(docs)} Retrieved Sources", expanded=False):
                for i, doc in enumerate(docs, start=1):
                    page = doc.metadata.get("page", "N/A")
                    source = os.path.basename(doc.metadata.get("source", "PDF"))

                    st.markdown(f"""
                    <div style="background: linear-gradient(to right, #fff3cd, #fff9e6);
                                padding: 16px;
                                border-left: 4px solid #ffc107;
                                border-radius: 8px;
                                margin-bottom: 12px;">
                        <div style="margin-bottom: 8px;">
                            <span class="source-chip">📍 Source {i}</span>
                            <span class="source-chip">📄 {source}</span>
                            <span class="source-chip">📖 Page {page}</span>
                        </div>
                        <div style="color: #333; line-height: 1.6;">
                            {doc.page_content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Save to chat history
            sources_info = [
                {"page": doc.metadata.get("page", "N/A"), "content": doc.page_content}
                for doc in docs
            ]
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.text,
                "sources": sources_info
            })

elif query:
    st.warning("⚠️ Please upload a PDF file first to start asking questions.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Built with ❤️ using Streamlit, LangChain & Gemini"
    "</div>",
    unsafe_allow_html=True
)