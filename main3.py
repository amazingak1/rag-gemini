import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_huggingface import HuggingFaceEmbeddings

# Gemini SDK
from google import genai


# ---------------- ENV ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)


# ---------------- UI ----------------
st.set_page_config(page_title="Document RAG Chat")
st.title("Chat with your Document (PDF, DOCX, TXT)")

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Chat")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

uploaded_file = st.file_uploader(
    "Upload a PDF, DOCX, or TXT file",
    type=["pdf", "docx", "txt"]
)


# ---------------- EMBEDDINGS (CACHE) ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()


# ---------------- VECTORSTORE ----------------
@st.cache_resource
def build_vectorstore(file_path: str, file_ext: str):

    file_ext = file_ext.lower().lstrip(".")

    if file_ext == "pdf":
        loader = PyPDFLoader(file_path)

    elif file_ext == "docx":
        loader = Docx2txtLoader(file_path)

    elif file_ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


vectorstore = None


# ---------------- LOAD FILE ----------------
if uploaded_file:

    file_ext = os.path.splitext(uploaded_file.name)[1]

    if not file_ext:
        st.error("Could not detect file extension.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        vectorstore = build_vectorstore(file_path, file_ext)
        st.success("Document indexed successfully!")

    except Exception as e:
        st.error(f"Error processing document: {e}")
        st.stop()


# ---------------- DISPLAY CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------------- QUERY INPUT ----------------
query = st.chat_input("Ask a question from the document...")

if query and vectorstore:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching..."):

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)

            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know (based on the provided document)".
Answer briefly.

Context:
{context}

Question:
{query}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            answer = response.text

            st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        # -------- SOURCES --------
        with st.expander("View Sources"):

            for i, doc in enumerate(docs, start=1):

                page = doc.metadata.get("page", "N/A")
                source = os.path.basename(
                    doc.metadata.get("source", "Document")
                )

                st.markdown(
                    f"""
                    **Source {i}: {source} (Page {page})**

                    {doc.page_content}

                    ---
                    """
                )


elif query:
    st.warning("Please upload a document first.")
