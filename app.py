import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------------
# Gemini setup
# ------------------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------------
# Load vector DB
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="My Personal RAG Bot")

st.title("📄 My Personal RAG Chatbot (Gemini)")

query = st.text_input("Ask a question from my personal document")

if query:
    docs = db.similarity_search(query, k=3)

    context = ""
    for i, d in enumerate(docs):
        context += f"Document part {i+1}:\n{d.page_content}\n\n"

    prompt = f"""
Answer the question ONLY using the information below.

If the answer is not present in the data, say:
"I do not have this information in my document."

Information:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)

    st.subheader("Answer")
    st.write(response.text)
