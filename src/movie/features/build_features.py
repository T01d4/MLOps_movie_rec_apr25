import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Get ENV variables or use fallbacks
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
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
    # Target paths
    movie_matrix_path = os.path.join(PROCESSED_DIR, "movie_matrix.csv")
    user_matrix_path = os.path.join(PROCESSED_DIR, "user_matrix.csv")

    # Only build if files are missing or force_rebuild=True
    if (not os.path.exists(movie_matrix_path) or
        not os.path.exists(user_matrix_path) or
        force_rebuild):

        print("🔄 Building feature files ...")
        ratings = read_ratings("ratings.csv")
        movies = read_movies("movies.csv")
        user_matrix = create_user_matrix(ratings, movies)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        movies_no_title = movies.drop(columns=["title"])
        movies_no_title.to_csv(movie_matrix_path, index=False)
        user_matrix.to_csv(user_matrix_path)
        print("✅ Feature files created!")
    else:
        print("✅ Feature files already exist. No build required.")

if __name__ == "__main__":
    main()