import os
from utils.pdf_utils import load_company_documents
from utils.embedding_utils import chunk_text
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_FOLDER = "data"
EMBEDDINGS_FOLDER = "embeddings"

def build_and_save_index_for_company(company_name, pdf_file):
    # Load PDF text
    text = load_company_documents(pdf_file, DATA_FOLDER)
    if not text.strip():
        print(f"No text found in the document for {company_name}. Skipping.")
        return
    
    # Chunk text
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    if not chunks:
        print(f"No chunks generated for {company_name}. Check the chunking logic.")
        return
    
    # Log chunk details
    print(f"Chunks for {company_name}: {len(chunks)} chunks created.")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and save FAISS index using LangChain
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(EMBEDDINGS_FOLDER, index_name=company_name)
    
    print(f"Successfully saved index for {company_name}")

def main():
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

    # Map real company names to their PDF file names
    companies = {
        "Eicher": "Eicher.pdf",
        "KRN": "KRN.pdf",
        "Zomato": "Zomato.pdf",
    }

    for company_name, pdf_file in companies.items():
        print(f"Processing {company_name} ({pdf_file})...")
        build_and_save_index_for_company(company_name, pdf_file)
    print("Preprocessing completed. Indices are built.")

if __name__ == "__main__":
    main()
