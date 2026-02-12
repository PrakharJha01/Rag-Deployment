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
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Personal RAG Bot")

st.title("📄 Personal RAG Chatbot (Gemini)")

# ------------------------
# Upload personal file
# ------------------------

@st.cache_resource
def load_rag_db():

    with open("my_document.txt", "r", encoding="utf-8") as f:
        text = f.read()

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


db = load_rag_db()

st.success("Personal document loaded.")

# -------------------------
# Chat
# -------------------------
query = st.text_input("Ask a question from my personal document")

if query:
    docs = db.similarity_search(query, k=3)

    context = ""
    for d in docs:
        context += d.page_content + "\n\n"

    prompt = f"""
Answer the question only using the information below.

If the answer is not present, say:
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
