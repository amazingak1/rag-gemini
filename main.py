import streamlit as st
import os
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RAG with Gemini")
st.title("📄 RAG Document QA (Gemini 2.5 Flash)")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ---------------- LOAD & EMBED ----------------
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# ---------------- GEMINI MODEL ----------------
model = genai.GenerativeModel("gemini-2.5-flash")

query = st.text_input("Ask a question from the document")

if query:
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)
    st.subheader("Answer")
    st.write(response.text)
