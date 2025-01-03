import os
import PyPDF2

def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        with open(pdf_file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_file_path}: {e}")
    return text

def load_company_documents(pdf_file, data_folder="data"):
    """
    Loads all PDFs for the given company from the data folder.
    """
    pdf_path = os.path.join(data_folder, pdf_file)
    if not os.path.exists(pdf_path):
        print(f"File {pdf_file} not found in {data_folder}.")
        return ""
    return extract_text_from_pdf(pdf_path)
