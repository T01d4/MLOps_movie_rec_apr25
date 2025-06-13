import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Hole ENV-Variablen oder nimm Fallbacks
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def read_ratings(ratings_csv, data_dir=RAW_DIR) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    temp = pd.DataFrame(LabelEncoder().fit_transform(data["movieId"]))
    data["movieId"] = temp
    return data

def read_movies(movies_csv, data_dir=RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    genres = df["genres"].str.get_dummies(sep="|")
    result_df = pd.concat([df[["movieId", "title"]], genres], axis=1)
    return result_df

def create_user_matrix(ratings, movies):
    movie_ratings = ratings.merge(movies, on="movieId", how="inner")
    movie_ratings = movie_ratings.drop(["movieId", "timestamp", "title", "rating"], axis=1)
    user_matrix = movie_ratings.groupby("userId").agg("mean")
    return user_matrix

def main(force_rebuild=False):
    # Zielpfade
    movie_matrix_path = os.path.join(PROCESSED_DIR, "movie_matrix.csv")
    user_matrix_path = os.path.join(PROCESSED_DIR, "user_matrix.csv")

    # Nur bauen, wenn Files fehlen oder force_rebuild=True
    if (not os.path.exists(movie_matrix_path) or
        not os.path.exists(user_matrix_path) or
        force_rebuild):

        print("🔄 Baue Feature-Files ...")
        ratings = read_ratings("ratings.csv")
        movies = read_movies("movies.csv")
        user_matrix = create_user_matrix(ratings, movies)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        movies_no_title = movies.drop(columns=["title"])
        movies_no_title.to_csv(movie_matrix_path, index=False)
        user_matrix.to_csv(user_matrix_path)
        print("✅ Feature-Files erzeugt!")
    else:
        print("✅ Feature-Files bereits vorhanden. Kein Build nötig.")

if __name__ == "__main__":
    main()