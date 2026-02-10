import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# -------- LOAD ENV --------
# BASE_DIR = Path(__file__).resolve().parent
# ENV_PATH = BASE_DIR / ".env"
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found")

# ✅ CRITICAL FIX (force SDK to see the key)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# genai.configure(api_key=GOOGLE_API_KEY)

# Create client AFTER configure
client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------- UI HIGHLIGHT CONFIG ----------------
HIGHLIGHT_BG_COLOR = "#fff3cd"
HIGHLIGHT_BORDER_COLOR = "#ffcc00"

# ---------------- CHUNKING CONFIG ----------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------------- RETRIEVAL CONFIG ----------------
TOP_K = 3

# ---------------- MODEL CONFIG ----------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------- PROMPT TEMPLATE ----------------
PROMPT_TEMPLATE = """
Answer ONLY using the context below.
If the answer is not present, say "I don't know".
Try to answer in short.

Context:
{context}

Question:
{question}
"""
#streamlit run app.py