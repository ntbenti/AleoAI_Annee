import os
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

if not GITHUB_TOKEN:
    raise ValueError("GitHub token not found. Please add it to the .env file.")

# Use 'Bearer' or 'token' depending on your token type
headers = {
    'Authorization': f'Bearer {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_repository_info(repo_full_name):
    url = f'https://api.github.com/repos/{repo_full_name}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        repo = response.json()
        return {
            'full_name': repo['full_name'],
            'clone_url': repo['clone_url']
        }
    else:
        print(f"Failed to fetch repository {repo_full_name}: {response.status_code} {response.reason}")
        if response.status_code == 403:
            print("Check if your GitHub token has the necessary permissions and has not exceeded rate limits.")
        elif response.status_code == 404:
            print("Repository not found. Please check the repository name in repos.txt.")
        return None

def main():
    # Read repository names from repos.txt
    if not os.path.exists('repos.txt'):
        raise FileNotFoundError("repos.txt file not found. Please create it and list repository names.")

    with open('repos.txt', 'r') as f:
        repo_full_names = [line.strip() for line in f if line.strip()]

    if not repo_full_names:
        print("No repository names found in repos.txt.")
        return

    print("Collecting repositories...")
    repos_data = []
    for repo_full_name in tqdm(repo_full_names, desc="Fetching Repos"):
        repo_info = get_repository_info(repo_full_name)
        if repo_info:
            repos_data.append(repo_info)

    # Save repository info to CSV
    if repos_data:
        df = pd.DataFrame(repos_data)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/repositories.csv', index=False)
        print("Repositories saved to data/repositories.csv")
    else:
        print("No repository information collected.")

if __name__ == "__main__":
    main()