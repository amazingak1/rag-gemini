### 5️⃣ `rag.py` — Retrieval + Generation Logic

**Purpose:**
- Core **RAG pipeline**

**Responsibilities:**
- Retrieve top-k relevant chunks from FAISS
- Build prompt using retrieved context
- Call Gemini model
- Return:
  - Final answer
  - Retrieved documents (sources)

**Keeps AI logic isolated from UI**

---

### 6️⃣ `requirements.txt` — Dependencies

**Purpose:**
- Lists all required Python packages

**Install dependencies:**
