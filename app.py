import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------------
# Gemini setup
# ------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

st.set_page_config(page_title="Personal RAG Bot")

st.title("📄 Personal RAG Chatbot (Gemini)")

# ------------------------
# Upload personal file
# ------------------------
uploaded_file = st.file_uploader(
    "Upload your personal 2-page document (txt file)",
    type=["txt"]
)

@st.cache_resource
def create_db(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)
    return db


db = None

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    db = create_db(text)
    st.success("Document indexed successfully!")

# ------------------------
# Chat
# ------------------------
if db:
    query = st.text_input("Ask a question from your document")

    if query:
        docs = db.similarity_search(query, k=3)

        context = ""
        for d in docs:
            context += d.page_content + "\n\n"

        prompt = f"""
Answer the question using only the information below.

If the answer is not found, say:
"I do not have this information in my document."

Information:
{context}

Question:
{query}
"""

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )

        st.subheader("Answer")
        st.write(response.text)
