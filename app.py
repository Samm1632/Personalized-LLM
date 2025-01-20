import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "Provide your key here"

st.title("SkyNet")

# Directory for ChromaDB persistence
persist_directory = "./chromadb_pdf_store"

# Initialize variables
vectordb = None  # To store the ChromaDB instance

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    try:
        # Process PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store in ChromaDB
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectordb = Chroma.from_texts(
            texts=[chunk.page_content for chunk in chunks],
            embedding=embeddings,
            persist_directory=persist_directory
        )

        st.success("PDF processed and stored in ChromaDB!")
    except Exception as e:
        st.error(f"Error: {e}")

# Query Option
if vectordb:
    st.header("We are ready to go!")
    query = st.text_input("Launch your query:")
    if query:
        try:
            # Initialize the QA chain
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            
            # Retrieve relevant documents
            relevant_docs = vectordb.similarity_search(query, k=5)
            
            # Generate an answer
            answer = qa_chain.run(input_documents=relevant_docs, question=query)
            st.write(f"**Answer:**  {answer}")
        except Exception as e:
            st.error(f"Error while querying: {e}")