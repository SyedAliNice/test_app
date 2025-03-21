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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


# Load API keys from Streamlit secrets
groq_api = st.secrets["GROQ_API_KEY"]

st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

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
    
    
    # Create vector store and retriever
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever()
    
    # Initialize LLM
    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api)
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that will make the question paper according to the provided document's context with different requirements like number of 
        short questoin, 
        long questions,
        Multiple Choice Questions,
        and the difficultly level.
        Make the question paper from the provided documents but don't write questions as it is rephrase the questions according to the difficulty level.
        As well as whenever you are asked to write everything in urdu please make sure that you don't make grammatical mistakes in it."""),
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
