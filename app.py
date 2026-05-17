import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Study Buddy RAG")
st.title("📚 Study Buddy RAG")
st.write("Upload your notes/PDF and ask questions!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded!")

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    If you don't know, say "I don't know based on the document."
    
    Context: {context}
    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("❓ Ask a question:")
    if query:
        answer = chain.invoke(query)
        if "i don't know" in answer.lower():
            st.error("❌ Answer not found in document.")
        else:
            st.success(answer)
