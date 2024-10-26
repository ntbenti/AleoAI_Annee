import os
import PyPDF2
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            try:
                text += page.extract_text() or ''
            except Exception as e:
                print(f"Error extracting text from page: {e}")
    return text

def main():
    pdfs_dir = 'pdfs'
    output_dir = 'data/pdf_texts'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]

    print("Extracting text from PDFs...")
    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        output_file = os.path.splitext(pdf_file)[0] + '.txt'
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

    print("Finished extracting text from PDFs.")

if __name__ == "__main__":
    main()