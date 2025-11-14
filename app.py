# app.py
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import tempfile
from typing import List

# App configuration
st.set_page_config(page_title="PDF Chatbot — Mini RAG (CPU)", layout="wide")
st.title("PDF Chatbot — Mini RAG (CPU-friendly)")
st.write("Upload a PDF and ask questions about its content. The system uses local models and FAISS for retrieval.")

# -----------------------
# Helper functions
# -----------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file and return as a single string."""
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()

def build_vectorstore(text: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Split text into chunks, compute embeddings, and create a FAISS vectorstore.
    Returns: (vectorstore, docs) where docs is a list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs: List[str] = splitter.split_text(text)
    embed = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_texts(texts=docs, embedding=embed)
    return vectorstore, docs

def generate_answer_with_pipeline(query: str, contexts: List[str], llm_pipeline, max_length: int = 256) -> str:
    """
    Build a prompt using the top contexts and generate an answer with the LLM pipeline.
    If the answer is not in the contexts, the prompt instructs the model to reply 'I don't know'.
    """
    context_text = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Use only the context provided below to answer the question. "
        "If the answer is not contained in the context, reply exactly: I don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely:"
    )
    output = llm_pipeline(prompt, max_length=max_length, do_sample=False)
    # pipeline returns list of dicts; many models use 'generated_text'
    first = output[0]
    answer = first.get("generated_text") or first.get("summary_text") or str(first)
    return answer.strip()

# -----------------------
# UI: upload and processing
# -----------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if not uploaded_file:
    st.info("Upload a PDF to begin. Use a short PDF (1–10 pages) for a quick demo.")
    st.stop()

st.write("Processing uploaded PDF...")
with st.spinner("Extracting text..."):
    raw_text = extract_text_from_pdf(uploaded_file)

if not raw_text:
    st.error("No text could be extracted from the PDF. If your PDF is scanned, run OCR first.")
    st.stop()

st.success(f"PDF text extracted. Document length: {len(raw_text)} characters.")

# Use a cache key so we do not re-create embeddings repeatedly for the same file
cache_key = f"vec_{uploaded_file.name}_{uploaded_file.size}"

if cache_key not in st.session_state:
    with st.spinner("Creating embeddings and building FAISS index (this may take a while)..."):
        vectorstore, docs = build_vectorstore(raw_text)
        st.session_state[cache_key] = {"vectorstore": vectorstore, "docs": docs}
        st.success("Index built and stored in session.")
else:
    vectorstore = st.session_state[cache_key]["vectorstore"]
    docs = st.session_state[cache_key]["docs"]

# -----------------------
# Model selection and loading
# -----------------------
st.sidebar.header("Model settings")
model_name = st.sidebar.selectbox("Select a CPU-friendly model", ["google/flan-t5-small", "google/flan-t5-base"])
max_tokens = st.sidebar.slider("Max tokens for answer", 64, 512, 256, step=64)
retrieval_k = st.sidebar.slider("Number of chunks to retrieve (k)", 1, 8, 4)

@st.cache_resource
def load_pipeline(model_name: str):
    """Load transformers text2text pipeline on CPU (device=-1)."""
    return pipeline("text2text-generation", model=model_name, device=-1)

with st.spinner("Loading text generation model..."):
    llm_pipeline = load_pipeline(model_name)

# -----------------------
# Question UI and answering
# -----------------------
st.subheader("Ask a question about the uploaded PDF")
query = st.text_input("Enter your question here")

if st.button("Get Answer") and query.strip():
    with st.spinner("Retrieving relevant chunks..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
        retrieved_docs = retriever.get_relevant_documents(query)
        contexts = [d.page_content for d in retrieved_docs]

    with st.spinner("Generating answer..."):
        answer = generate_answer_with_pipeline(query, contexts, llm_pipeline, max_length=max_tokens)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Retrieved Chunks (sources)")
    for i, d in enumerate(retrieved_docs, start=1):
        snippet = d.page_content[:400] + ("..." if len(d.page_content) > 400 else "")
        st.write(f"Chunk {i}: {snippet}")
else:
    st.info("Type a question and press 'Get Answer' to query the PDF.")


