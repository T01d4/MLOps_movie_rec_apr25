import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging
import shutil

def safe_write_csv(df: pd.DataFrame, target_path: str, logger: logging.Logger):
    tmp_path = f"{target_path}.tmp"
    try:
        df.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, target_path)  # ersetzt ggf. vorhandene Datei
        logger.info(f"Datei erfolgreich gespeichert unter: {target_path}")
    except Exception as e:
        logger.error(f"Fehler beim Schreiben der Datei {target_path}: {e}")
        raise

def read_ratings(ratings_csv, data_dir="data/raw") -> pd.DataFrame:
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    temp = pd.DataFrame(LabelEncoder().fit_transform(data["movieId"]))
    data["movieId"] = temp
    return data

def read_movies(movies_csv, data_dir="data/raw") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    genres = df["genres"].str.get_dummies(sep="|")
    result_df = pd.concat([df[["movieId", "title"]], genres], axis=1)
    return result_df

def create_user_matrix(ratings, movies):
    movie_ratings = ratings.merge(movies, on="movieId", how="inner")
    for col in ["movieId", "timestamp", "title", "rating"]:
        if col in movie_ratings.columns:
            movie_ratings = movie_ratings.drop(col, axis=1)
    user_matrix = movie_ratings.groupby("userId").agg("mean")
    return user_matrix

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Lese ratings und movies ein...")

    user_ratings = read_ratings("ratings.csv", data_dir="data/raw")
    movies = read_movies("movies.csv", data_dir="data/raw")

    logger.info("Erstelle user-movie matrix...")
    user_matrix = create_user_matrix(user_ratings, movies)

    movies = movies.drop("title", axis=1)

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    movie_matrix_path = os.path.join(output_dir, "movie_matrix.csv")
    user_matrix_path = os.path.join(output_dir, "user_matrix.csv")

    logger.info(f"Speichere movie_matrix nach: {movie_matrix_path}")
    safe_write_csv(movies, movie_matrix_path, logger)

    logger.info(f"Speichere user_matrix nach: {user_matrix_path}")
    safe_write_csv(user_matrix.reset_index(), user_matrix_path, logger)  # <--- Anpassung hier

    logger.info("Feature-Erstellung abgeschlossen.")

if __name__ == "__main__":
    main()