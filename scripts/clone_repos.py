import os
import pandas as pd
from git import Repo
from tqdm import tqdm

def main():
    df = pd.read_csv('data/repositories.csv')
    repos_dir = 'data/cloned_repos'

    if not os.path.exists(repos_dir):
        os.makedirs(repos_dir)

    print("Cloning repositories...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Cloning Repos"):
        repo_name = row['full_name'].replace('/', '_')
        clone_url = row['clone_url']
        repo_path = os.path.join(repos_dir, repo_name)
        if not os.path.exists(repo_path):
            try:
                Repo.clone_from(clone_url, repo_path)
            except Exception as e:
                print(f"Failed to clone {clone_url}: {e}")
        else:
            print(f"Repository {repo_name} already cloned.")

    print("Finished cloning repositories.")

if __name__ == "__main__":
    main()