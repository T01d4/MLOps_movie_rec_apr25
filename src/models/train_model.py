import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import mlflow
import mlflow.sklearn

def train_model(user_matrix):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        user_matrix.drop("userId", axis=1)
    )
    return nbrs

def main(input_filepath="data/processed", model_path="models/model.pkl"):
    user_matrix_path = os.path.join(input_filepath, "movies_matrix.csv")

    mlflow.set_experiment("movie_recommendation")
    with mlflow.start_run(run_name="knn_model"):

        try:
            print(f"Lade Daten aus {user_matrix_path}")
            user_matrix = pd.read_csv(user_matrix_path)

            model = train_model(user_matrix)

            mlflow.log_param("n_neighbors", 20)
            mlflow.log_param("algorithm", "ball_tree")

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as filehandler:
                pickle.dump(model, filehandler)

            print(f"Modell gespeichert unter: {model_path}")
            mlflow.log_artifact(model_path)

        except Exception as e:
            print(f"Fehler beim Training: {e}")
            mlflow.log_param("status", "failed")

if __name__ == "__main__":
    main()