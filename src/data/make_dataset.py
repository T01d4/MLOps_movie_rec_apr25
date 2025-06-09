# src/data/make_dataset.py
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from tqdm import tqdm

# === ENV laden ===
load_dotenv(find_dotenv())

# Hole ENV-Variablen oder nutze Fallbacks
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def main(input_filepath=RAW_DIR, output_filepath=PROCESSED_DIR):
    logger = logging.getLogger(__name__)
    logger.info('📦 Starte Verarbeitung der Rohdaten.')

    movie_matrix_path = os.path.join(output_filepath, "movies_matrix.csv")

    # Falls Datei existiert: Verarbeitung überspringen
    if os.path.exists(movie_matrix_path):
        logger.info("✅ movies_matrix.csv bereits vorhanden – Verarbeitung wird übersprungen.")
        return

    scores_path = os.path.join(input_filepath, "genome-scores.csv")
    ratings_path = os.path.join(input_filepath, "ratings.csv")

    df_scores = pd.read_csv(scores_path, dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})
    df_ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"],
                             dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

    movie_embeddings = df_scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

    user_vectors = []
    user_ids = []

    for user_id, group in tqdm(df_ratings.groupby("userId")):
        rated_movies = group["movieId"].values
        common_movies = [mid for mid in rated_movies if mid in movie_embeddings.index]
        if not common_movies:
            continue
        vectors = movie_embeddings.loc[common_movies]
        user_vector = vectors.mean(axis=0)
        user_vectors.append(user_vector)
        user_ids.append(user_id)

    movies_matrix = pd.DataFrame(user_vectors, index=user_ids)
    movies_matrix.index.name = "userId"

    os.makedirs(output_filepath, exist_ok=True)
    movies_matrix.reset_index(inplace=True)
    movies_matrix.to_csv(movie_matrix_path, index=False, encoding='utf-8')
    logger.info(f"✅ movies_matrix.csv gespeichert unter: {movie_matrix_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()