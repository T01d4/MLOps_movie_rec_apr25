import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os

import mlflow
import mlflow.sklearn


def train_model(movie_matrix):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs


if __name__ == "__main__":
    movie_matrix_path = "data/processed/movie_matrix.csv"
    model_path = "models/model.pkl"

    mlflow.set_experiment("movie_recommendation")
    with mlflow.start_run(run_name="knn_model"):

        try:
            print(f"Lade Daten aus {movie_matrix_path}")
            movie_matrix = pd.read_csv(movie_matrix_path)

            model = train_model(movie_matrix)

            mlflow.log_param("n_neighbors", 20)
            mlflow.log_param("algorithm", "ball_tree")

            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as filehandler:
                pickle.dump(model, filehandler)

            print(f"Modell gespeichert unter: {model_path}")
            mlflow.log_artifact(model_path)

        except Exception as e:
            print(f"Fehler beim Training: {e}")
            mlflow.log_param("status", "failed")