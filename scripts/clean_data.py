import json
import re
from tqdm import tqdm

def clean_text(text):
    # Remove code comments and unnecessary whitespace
    # This is a simplified example; you may need more robust cleaning
    text = re.sub(r'//.*', '', text)  # Remove single-line comments (e.g., C++, Java)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)  # Remove multi-line comments (e.g., C++, Java)
    text = re.sub(r'#.*', '', text)  # Remove Python and shell comments
    text = re.sub(r'<!--[\s\S]*?-->', '', text)  # Remove HTML comments
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    return text.strip()

def main():
    with open('data/processed/final_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = {}

    print("Cleaning data...")
    for key, files in tqdm(data.items(), desc="Datasets"):
        cleaned_files = {}
        for file_name, content in files.items():
            cleaned_content = clean_text(content)
            cleaned_files[file_name] = cleaned_content
        cleaned_data[key] = cleaned_files

    # Save cleaned data
    with open('data/processed/cleaned_data.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4)

    print("Cleaned data saved to data/processed/cleaned_data.json")

if __name__ == "__main__":
    main()