import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

import os
os.environ["OPENAI_API_KEY"] = "**************************************************************"

# Load the QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Load the prompt template
PROMPT_TEMPLATE = """You are an AI assistant. Your goal is to answer the questions as accurately as possible based on the query given by the user.


{query}"""

# Create the streamlit app
st.title("PDF Query Search")

# Section for uploading a PDF
pdf_uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if pdf_uploaded is not None:
    # Create the "uploaded_files" directory if it doesn't exist
    uploaded_files_dir = os.path.join("E:", "uploaded_files")
    if not os.path.exists(uploaded_files_dir):
        os.makedirs(uploaded_files_dir)

    # Save the uploaded file
    with open(os.path.join(uploaded_files_dir, pdf_uploaded.name), "wb") as f:
        f.write(pdf_uploaded.getbuffer())

    # Load the PDF file
    pdf_file = os.path.join(uploaded_files_dir, pdf_uploaded.name)
    pdfreader = PdfReader(pdf_file)

    # Extract the text from the PDF file
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Initialize the text splitter and embeddings
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()

    # Create the FAISS document search
    document_search = FAISS.from_texts(texts, embeddings)

# Section for asking questions
query = st.text_input("Your Question:: ")

if query:
    docs = document_search.similarity_search(query)

    response = chain.run(input_documents=docs, question=query, prompt_template=PROMPT_TEMPLATE)

    st.write(response)