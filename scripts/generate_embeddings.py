import json
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    with open('data/cleaned_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = {}
    for key, files in data.items():
        for file_name, content in files.items():
            embedding = model.encode(content)
            embeddings[f"{key}/{file_name}"] = embedding.tolist()

    # Save embeddings
    with open('data/embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)

    print("Embeddings generated and saved to data/embeddings.json")

if __name__ == "__main__":
    main()