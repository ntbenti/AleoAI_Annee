import json
import os

def main():
    # Path to the cleaned data JSON file
    cleaned_data_path = 'data/processed/cleaned_data.json'

    # Output path for the training data
    training_data_path = 'data/processed/training_data.txt'

    # Ensure the output directory exists
    os.makedirs('data/processed', exist_ok=True)

    # Read the cleaned data
    with open(cleaned_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect all text content
    texts = []

    for dataset_name, files in data.items():
        for file_name, content in files.items():
            texts.append(content.strip())

    # Write the texts to the training data file
    with open(training_data_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    print(f"Training data saved to {training_data_path}")

if __name__ == "__main__":
    main()