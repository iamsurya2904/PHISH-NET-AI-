from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import load_chain
import os

# Error handling for Streamlit import
try:
    import streamlit as st
except ImportError as e:
    print(f"Streamlit is not installed: {e}")
    st = None

# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Initialize LLM and chains once
@st.cache_resource
def initialize_llm():
    return OpenAI(temperature=0)

@st.cache_resource
def create_summary_chain(llm):
    return load_chain("summarization", llm=llm)

@st.cache_resource
def create_qa_chain(llm):
    return load_chain("question-answering", llm=llm)

def summarize_with_langchain(file):
    """
    Summarizes the given document using LangChain.
    Args:
        file: The uploaded document file.
    Returns:
        str: A summary of the document.
    """
    try:
        # Load the document
        loader = UnstructuredFileLoader(file)
        documents = loader.load()

        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)

        # Initialize LLM and create a summarization chain
        llm = initialize_llm()
        chain = create_summary_chain(llm)

        summary = chain.run(input_documents=texts)
        return summary
    except Exception as e:
        return f"Error during summarization: {e}"

def qa_with_langchain(question, context):
    """
    Performs QA on the provided context using LangChain.
    Args:
        question (str): The user's question.
        context (str): The context (e.g., summary of a document).
    Returns:
        str: The answer to the user's question.
    """
    try:
        # Initialize LLM and create a QA chain
        llm = initialize_llm()
        chain = create_qa_chain(llm)

        answer = chain.run(input_documents=[context], question=question)
        return answer
    except Exception as e:
        return f"Error during QA: {e}"
