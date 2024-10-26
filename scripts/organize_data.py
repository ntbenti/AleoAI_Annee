import os
import json
from tqdm import tqdm

def read_code_files(repo_path):
    code_data = {}
    code_extensions = (
        '.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.c', '.cc', '.cxx',
        '.hpp', '.h', '.hxx', '.rs', '.go', '.sh', '.rb', '.swift', '.kt', '.scala',
        '.cs', '.php', '.html', '.css', '.json', '.xml', '.dart', '.pl', '.erl',
        '.ex', '.hs', '.leo', '.md', '.yml', '.toml', '.aleo'
    )
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(code_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    # Store content with relative path
                    relative_path = os.path.relpath(file_path, repo_path)
                    code_data[relative_path] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return code_data

def read_pdf_texts(pdf_dir):
    pdf_data = {}
    for file in os.listdir(pdf_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(pdf_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                pdf_name = os.path.splitext(file)[0]
                pdf_data[pdf_name] = content
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pdf_data

def main():
    data = {}

    # Process cloned repositories
    repos_dir = 'data/cloned_repos'
    repo_folders = [folder for folder in os.listdir(repos_dir) if os.path.isdir(os.path.join(repos_dir, folder))]

    print("Organizing code files from repositories...")
    for repo_folder in tqdm(repo_folders, desc="Repositories"):
        repo_path = os.path.join(repos_dir, repo_folder)
        code_files = read_code_files(repo_path)
        data[repo_folder] = code_files

    # Process PDF texts
    pdf_texts_dir = 'data/pdf_texts'
    pdf_texts = read_pdf_texts(pdf_texts_dir)
    data['pdf_texts'] = pdf_texts

    # Save data to JSON file
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/final_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print("Data organized and saved to data/processed/final_data.json")

if __name__ == "__main__":
    main()