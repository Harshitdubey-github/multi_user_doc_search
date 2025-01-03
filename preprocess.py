import os
from utils.pdf_utils import load_company_documents
from utils.embedding_utils import EmbeddingIndex, chunk_text
import faiss

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

    # Build index
    embedding_index = EmbeddingIndex()
    embedding_index.build_index(chunks)

    # Save FAISS index
    faiss_index_path = os.path.join(EMBEDDINGS_FOLDER, f"{company_name}.index")
    faiss.write_index(embedding_index.index, faiss_index_path)

    # Save doc_texts as well (for retrieval)
    with open(os.path.join(EMBEDDINGS_FOLDER, f"{company_name}_texts.txt"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")

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
