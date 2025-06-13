import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from dotenv import load_dotenv, find_dotenv

# === ENV laden ===
load_dotenv(find_dotenv())
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def train_model(movie_matrix):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs

if __name__ == "__main__":
    movie_matrix_path = os.path.join(DATA_DIR, "processed", "movie_matrix.csv")
    model_path = os.path.join(MODEL_DIR, "model.pkl")

    movie_matrix = pd.read_csv(movie_matrix_path)
    model = train_model(movie_matrix)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as filehandler:
        pickle.dump(model, filehandler)