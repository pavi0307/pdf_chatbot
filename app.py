# app.py
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from typing import List

# -----------------------------------
# Streamlit App Config
# -----------------------------------
st.set_page_config(page_title="PDF Chatbot — Mini RAG (CPU)", layout="wide")
st.title(" PDF Chatbot — Mini RAG ")
st.write("Upload a PDF and ask questions about its content. The system uses embeddings + FAISS + a CPU-friendly text generation model.")

# -----------------------------------
# Helper: Extract PDF text
# -----------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


# -----------------------------------
# Helper: Build vectorstore
# -----------------------------------
def build_vectorstore(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks: List[str] = splitter.split_text(text)

    embed_model = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(chunks, embed_model)

    return vectorstore, chunks


# -----------------------------------
# Helper: Generate answer using pipeline
# -----------------------------------
def generate_answer(query: str, contexts: List[str], llm, max_length: int = 256) -> str:
    context_text = "\n\n---\n\n".join(contexts)

    prompt = (
        "You are a helpful AI assistant. Use ONLY the context below to answer the question. "
        "If the answer is not in the context, reply exactly with: I don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely:"
    )

    output = llm(prompt, max_length=max_length, do_sample=False)
    first = output[0]
    answer = first.get("generated_text") or first.get("summary_text") or str(first)

    return answer.strip()


# -----------------------------------
# PDF Upload
# -----------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if not uploaded_file:
    st.info("Please upload a PDF to begin.")
    st.stop()

st.write("Processing PDF...")
with st.spinner("Extracting text..."):
    raw_text = extract_text_from_pdf(uploaded_file)

if not raw_text:
    st.error("No extractable text found. The PDF may be scanned; try an OCR PDF.")
    st.stop()

st.success(f"PDF loaded. Total characters: {len(raw_text)}")


# -----------------------------------
# Create vectorstore (cached per file)
# -----------------------------------
cache_key = f"vec_{uploaded_file.name}_{uploaded_file.size}"

if cache_key not in st.session_state:
    with st.spinner("Building embeddings + FAISS vectorstore..."):
        vectorstore, docs = build_vectorstore(raw_text)
        st.session_state[cache_key] = {"vs": vectorstore, "docs": docs}
        st.success("FAISS index ready.")
else:
    vectorstore = st.session_state[cache_key]["vs"]
    docs = st.session_state[cache_key]["docs"]


# -----------------------------------
# Sidebar: Model selection
# -----------------------------------
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Choose a CPU-friendly model",
    ["google/flan-t5-small", "google/flan-t5-base"]
)
max_tokens = st.sidebar.slider("Max output tokens", 64, 512, 256, step=64)
retrieval_k = st.sidebar.slider("Chunks to retrieve (k)", 1, 8, 4)

@st.cache_resource
def load_pipeline(model_name: str):
    return pipeline("text2text-generation", model=model_name, device=-1)

with st.spinner("Loading model..."):
    llm_pipeline = load_pipeline(model_name)


# -----------------------------------
# Ask Question
# -----------------------------------
st.subheader("Ask a question about the PDF")

query = st.text_input("Your question")

if st.button("Get Answer") and query.strip():
    # ---- Retrieval ----
    with st.spinner("Retrieving relevant chunks..."):
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}
        )

        # NEW API — avoids Pydantic errors
        retrieved_docs = retriever.invoke(query)

        contexts = [doc.page_content for doc in retrieved_docs]

    # ---- LLM Answer ----
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, contexts, llm_pipeline, max_length=max_tokens)

    # ---- Output ----
    st.markdown("###  Answer")
    st.write(answer)

    st.markdown("###  Retrieved Context Chunks")
    for i, doc in enumerate(retrieved_docs, start=1):
        preview = doc.page_content[:400]
        st.write(f"**Chunk {i}:** {preview}...")
else:
    st.info("Enter a question and click 'Get Answer'.")

