# ai-rag-pdf-chatbot
An AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents using FastAPI, FAISS, Sentence Transformers, and Google Gemini.

# AI Resume RAG Chatbot

An AI-powered chatbot that can answer questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF files
- Automatic text extraction
- Smart document chunking
- Semantic search using embeddings
- FAISS vector database
- Google Gemini integration
- REST APIs using FastAPI
- Source chunk retrieval

## Tech Stack

- Python
- FastAPI
- FAISS
- Google Gemini API
- Sentence Transformers
- LangChain
- PyPDF

## Project Architecture

PDF Upload
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embeddings
        ↓
FAISS Vector Store
        ↓
Similarity Search
        ↓
Gemini LLM
        ↓
Final Answer

## API Endpoints

POST /upload-pdf

Uploads and indexes a PDF.

POST /ask

Accepts a natural language question and returns an answer with relevant source chunks.

## Installation

pip install -r requirements.txt

uvicorn main:app --reload

Open

http://127.0.0.1:8000/docs
