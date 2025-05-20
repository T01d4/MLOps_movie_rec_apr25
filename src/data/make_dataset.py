import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from tqdm import tqdm


def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('📦 Starte Verarbeitung der Rohdaten.')

    user_matrix_path = os.path.join(output_filepath, "user_matrix.csv")
    movie_matrix_path = os.path.join(output_filepath, "movies_matrix.csv")

    # Falls beide existieren: Verarbeitung überspringen
    if os.path.exists(user_matrix_path) and os.path.exists(movie_matrix_path):
        logger.info("✅ Beide Dateien vorhanden – Verarbeitung wird übersprungen.")
        return

    input_files = {
        "scores": os.path.join(input_filepath, "genome-scores.csv"),
        "ratings": os.path.join(input_filepath, "ratings.csv")
    }

    process_data(input_files, output_filepath, logger)


def process_data(input_files, output_filepath, logger):
    try:
        logger.info("📥 Lade CSV-Dateien...")

        df_scores = pd.read_csv(input_files["scores"], dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})
        df_ratings = pd.read_csv(input_files["ratings"], usecols=["userId", "movieId", "rating"],
                                 dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

        os.makedirs(output_filepath, exist_ok=True)

        user_matrix_path = os.path.join(output_filepath, "user_matrix.csv")
        if not os.path.exists(user_matrix_path):
            df = generate_user_matrix(df_ratings, df_scores, output_filepath, logger)
        else:
            df = pd.read_csv(user_matrix_path)
            logger.info("📄 user_matrix aus Datei geladen")

        movie_matrix = df.set_index('userId')
        movie_matrix.columns = ['_'.join(map(str, col)) for col in movie_matrix.columns]
        movie_matrix.reset_index(inplace=True)

        movie_matrix_path = os.path.join(output_filepath, 'movies_matrix.csv')
        movie_matrix.to_csv(movie_matrix_path, index=False, encoding='utf-8')
        logger.info(f"✅ movie_matrix gespeichert unter: {movie_matrix_path}")

    except Exception as e:
        logger.error(f"❌ Fehler bei der Verarbeitung: {e}")
        raise


def generate_user_matrix(df_ratings, df_scores, output_filepath, logger):
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

    user_matrix = pd.DataFrame(user_vectors, index=user_ids)
    user_matrix.index.name = "userId"

    user_matrix_path = os.path.join(output_filepath, "user_matrix.csv")
    logger.info(f"💾 Speichere user_matrix unter: {user_matrix_path}")
    user_matrix.to_csv(user_matrix_path, encoding='utf-8')

    return user_matrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    load_dotenv(find_dotenv())
    main("data/raw", "data/processed")