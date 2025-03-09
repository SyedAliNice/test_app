#import os
#from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
#load_dotenv()

st.title("PDF Question Answering App")

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily (if PyMuPDFLoader requires a file path)
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and split the document
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    documents = text_splitter.split_documents(docs)
    
    openai_api = "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"
    
    # Create vector store and retriever
    embeddings = OpenAIEmbeddings(api_key=openai_api)
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever()
    
    # Initialize LLM
    groq_api = "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"
    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api)
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided document context and if you will not able to answer the question from file, answer it according to your intelligence. but keep everything brief"),
        ("user", "Context: {context}\nQuestion: {question}")
    ])
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template,
            "verbose": True
        }
    )
    
    # User inputs their question
    input_text = st.text_input("Enter your question:")
    
    if input_text:
        with st.spinner("Searching for answer..."):
            result = qa_chain({"query": input_text})
        
        st.subheader("Answer:")
        st.write(result["result"])
        
        # Optionally, display source documents
        # st.subheader("Sources:")
        # for i, doc in enumerate(result["source_documents"], 1):
        #     st.write(f"Source {i}:")
        #     st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
