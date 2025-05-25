import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pdfplumber

st.title("Chat with your PDF")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document")

if pdf and query:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    text = extract_text_from_pdf("temp.pdf")
    vectorstore = create_vector_store(text)
    qa_chain = get_qa_chain(vectorstore)

    answer = qa_chain.run(query)
    st.write("🤖", answer)

def get_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa

def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text