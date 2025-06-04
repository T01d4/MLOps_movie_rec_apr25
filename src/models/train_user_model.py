# src/models/train_user_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import os
import mlflow
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import argparse
from mlflow.tracking import MlflowClient

class UserKNNWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["knn_model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> list:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame!")
        if model_input.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Feature mismatch: Model expects {self.model.n_features_in_} features, "
                f"but input has {model_input.shape[1]}"
            )
        indices = self.model.kneighbors(model_input.values, return_distance=False)
        return indices.tolist()

def train_user_model(n_neighbors=10):
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("user_model_exp")  # <--- KLARER EXPERIMENT-NAME!

    data_dir = "/opt/airflow/data/processed"
    ratings_path = "/opt/airflow/data/raw/ratings.csv"
    matrix_path = f"{data_dir}/movie_matrix.csv"
    user_matrix_path = f"{data_dir}/user_matrix.csv"
    features_path = f"{data_dir}/user_matrix_features.txt"
    model_dir = "/opt/airflow/models"
    model_path = f"{model_dir}/user_model.pkl"

    ratings = pd.read_csv(ratings_path)
    movie_embeddings = pd.read_csv(matrix_path, index_col=0).sort_index()
    feature_names = movie_embeddings.columns.tolist()

    user_vectors, user_ids = [], []
    for uid, group in ratings.groupby("userId"):
        rated = group[group["movieId"].isin(movie_embeddings.index)]
        if rated.empty:
            continue
        user_vector = movie_embeddings.loc[rated["movieId"]].mean(axis=0)
        user_vectors.append(user_vector)
        user_ids.append(uid)

    user_df = pd.DataFrame(user_vectors, index=user_ids, columns=feature_names)
    user_df.to_csv(user_matrix_path)
    with open(features_path, "w") as f:
        f.write("\n".join(feature_names))

    X = user_df.values
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    model.fit(X)

    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with mlflow.start_run(run_name="train_user_model") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "train_user_model")
        mlflow.log_param("model_type", "user_knn")
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("algorithm", "ball_tree")
        mlflow.log_metric("n_users", len(user_df))
        mlflow.log_artifact(user_matrix_path, artifact_path="features")
        mlflow.log_artifact(features_path, artifact_path="features")
        mlflow.log_artifact(model_path, artifact_path="backup_model")
        signature = infer_signature(
            pd.DataFrame(X, columns=feature_names),
            model.kneighbors(X[:5])[1]
        )
        mlflow.pyfunc.log_model(
            artifact_path="user_knn_pyfunc",
            python_model=UserKNNWrapper(),
            artifacts={"knn_model": model_path},
            signature=signature,
            input_example=pd.DataFrame(X[:5], columns=feature_names),
            registered_model_name="user_model"  # <--- KLARE REGISTRY!
        )
        client = MlflowClient()
        model_name = "user_model"
        run_id = run.info.run_id
        model_version = None
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.run_id == run_id:
                model_version = v.version
                break
        if model_version:
            client.set_model_version_tag(model_name, model_version, "n_neighbors", str(n_neighbors))
            client.set_model_version_tag(model_name, model_version, "algorithm", "ball_tree")
            print(f"📝 Tags für Modellversion {model_version} gesetzt: n_neighbors={n_neighbors}")
        else:
            print("⚠️ Konnte Modellversion für Tagging nicht bestimmen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=10)
    args = parser.parse_args()
    train_user_model(n_neighbors=args.n_neighbors)