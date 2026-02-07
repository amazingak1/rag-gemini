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
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------- UI ----------------
st.set_page_config(page_title="PDF RAG with Highlighting")
st.title("📘 Chat with PDF (RAG + Highlighted Sources)3")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# ---------------- EMBEDDINGS (ONCE) ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTORSTORE ----------------
@st.cache_resource
def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    return FAISS.from_documents(chunks, embeddings)

vectorstore = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    vectorstore = build_vectorstore(pdf_path)
    st.success("PDF indexed successfully!")

# ---------------- QUERY ----------------
query = st.text_input("Ask a question from the PDF")

if query and vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know".
try to answer in short
Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # -------- ANSWER --------
    st.subheader("Answer")
    st.write(response.text)

    # -------- SOURCES --------
    st.subheader("Answer Sources (Highlighted)")
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "N/A")
        source = os.path.basename(doc.metadata.get("source", "PDF"))

        with st.expander(f"Source {i}: {source} (page {page})"):
            st.markdown(
                f"""
                <div style="background-color:#fff3cd;
                            padding:12px;
                            border-left:6px solid #ffcc00;">
                {doc.page_content}
                </div>
                """,
                unsafe_allow_html=True
            )

elif query:
    st.warning("Please upload a PDF first.")
#streamlit run main3.py