# src/data/import_raw_data.py

import requests
import os
import logging
import subprocess
import shutil
import getpass
from dotenv import load_dotenv, find_dotenv

# === ENV load ===
load_dotenv(find_dotenv())
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

def import_raw_data(raw_data_path, filenames, bucket_folder_url):
    """Import filenames from bucket_folder_url into raw_data_path."""
    os.makedirs(raw_data_path, exist_ok=True)
    for filename in filenames:
        output_file = os.path.join(raw_data_path, filename)
        if os.path.exists(output_file):
            logging.info(f"✅ Data already exists, skip: {filename}")
            continue
        url = os.path.join(bucket_folder_url, filename)
        logging.info(f"⬇️  Download: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(output_file, "wb") as f:
                f.write(response.content)
            logging.info(f"✅ Data saved to : {output_file}")
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Download Error from {url}: {e}")

def main(raw_data_path=RAW_DIR,
         filenames=None,
         bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"):

    if filenames is None:
        filenames = [
            "genome-scores.csv", "genome-tags.csv", "links.csv",
            "movies.csv", "ratings.csv", "README.txt", "tags.csv"
        ]

    import_raw_data(raw_data_path, filenames, bucket_folder_url)
    logging.info("✅ Raw Data created.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()