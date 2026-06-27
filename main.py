from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import google.generativeai as genai
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# =========================
# GEMINI API KEY
# =========================

genai.configure(api_key="your_gemini_key")

model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# GLOBAL VECTOR STORE
# =========================

vector_db = None


# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():
    return {"message": "RAG Chat API Running"}


# =========================
# UPLOAD PDF
# =========================

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db

    contents = await file.read()

    # Extract text from PDF
    pdf = fitz.open(stream=contents, filetype="pdf")

    text = ""

    for page in pdf:
        text += page.get_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vector_db = FAISS.from_texts(chunks, embeddings)

    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "first_chunk": chunks[0] if chunks else ""
    }


# =========================
# QUESTION MODEL
# =========================

class QuestionRequest(BaseModel):
    question: str


# =========================
# ASK QUESTION
# =========================

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global vector_db

    if vector_db is None:
        return {
            "error": "Please upload a PDF first."
        }

    docs = vector_db.similarity_search(
        request.question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{request.question}
"""

    try:
        response = model.generate_content(prompt)

        return {
            "question": request.question,
            "answer": response.text,
            "sources": [doc.page_content[:200] for doc in docs]
        }

    except Exception as e:
        return {
            "question": request.question,
            "error": str(e),
            "sources": [doc.page_content[:200] for doc in docs]
        }