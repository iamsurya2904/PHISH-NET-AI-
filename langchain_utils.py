from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import load_chain
import os
import streamlit as st

# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

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
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("/tmp", file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

        # Load the document
        loader = UnstructuredFileLoader(temp_file_path)
        documents = loader.load()

        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)

        # Initialize LLM and create a summarization chain
        llm = initialize_llm()
        chain = create_summary_chain(llm)

        summary = chain.run(input_documents=texts)
        return summary
    except OSError as e:
        if "libGL.so.1" in str(e):
            return ("Error during summarization: libGL.so.1: cannot open shared object file: "
                    "No such file or directory. Please install the missing library using "
                    "'sudo apt-get install libgl1' or equivalent command for your OS.")
        return f"Error during summarization: {e}"
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
