import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber
import os
# Step 1: Load raw PDF(s)
all_text = ""
PDF_PATH = "data/Doctor_Help.pdf"  # Path to PDF file

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

print("Extracted text length:", len(all_text))

# Step 2: Create Chunks
def create_chunks(extracted_text):
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    # RecursiveCharacterTextSplitter expects a list of Documents, 
    # but we currently have a single string, so wrap it in a list
    from langchain.docstore.document import Document
    docs = [Document(page_content=extracted_text)]
    
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks

# Step 3: Generate chunks if text was extracted
if all_text.strip():  # Ensure text is not empty
    text_chunks = create_chunks(extracted_text=all_text)
    print("Number of text chunks:", len(text_chunks))
else:
    print("No text extracted from PDF.")


# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)