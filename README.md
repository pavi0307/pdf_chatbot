# PDF Chatbot — CPU-Friendly RAG

This Streamlit app allows users to chat with PDF documents using a lightweight Retrieval-Augmented Generation (RAG) pipeline.  
It runs entirely on CPU and uses open-source models from Hugging Face for embedding and text generation.

---

## Overview
The app extracts text from uploaded PDFs, splits the content into manageable chunks, creates vector embeddings, and retrieves the most relevant sections to answer user queries.  
It focuses on practicality and explainability, making it suitable for demonstrations or learning projects in AI and NLP.

---

## Features
- Extracts text from any PDF using `pypdf`
- Splits text into semantic chunks with `LangChain`
- Embeds text using `sentence-transformers/all-MiniLM-L6-v2`
- Stores and retrieves embeddings using FAISS
- Generates concise answers with `google/flan-t5-small` or `flan-t5-base`
- Fully CPU-compatible, no GPU or API key required

---

## Tech Stack
- **Framework:** Streamlit  
- **Language:** Python  
- **Libraries:** LangChain, Sentence Transformers, FAISS, Transformers, PyPDF  
- **Models:**  
  - Embedding: `sentence-transformers/all-MiniLM-L6-v2`  
  - LLM: `google/flan-t5-small` or `flan-t5-base`  

---

## Installation
Clone the repository and install dependencies:
```bash
git clone https://huggingface.co/spaces/Pavithra037/pdf-chatbot
cd pdf-chatbot
pip install -r requirements.txt
```

Then start the app:
```bash
streamlit run app.py
```

---

## Requirements
The following dependencies are required:
```
streamlit
langchain
langchain-community
transformers
pypdf
faiss-cpu
sentence-transformers
torch
```

---

## Usage
1. Upload a text-based PDF (preferably under 10 pages).
2. The app extracts and indexes the text into vector embeddings.
3. Ask a question related to the PDF content.
4. The model retrieves relevant chunks and generates an accurate answer.

---

## Example Queries
* "Who is mentioned in this resume?"
* "Summarize the key points from this report."
* "List all the projects mentioned in this document."

---

## Notes
* Works best on text-based PDFs (not scanned images).
* For scanned PDFs, OCR preprocessing is required.
* Designed for demonstration and educational use.

---

## Author
**Govindaswamy Pavithra**
