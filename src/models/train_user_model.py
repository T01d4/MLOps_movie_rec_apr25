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
import dagshub
# === .env laden ===
load_dotenv()
#dagshub.init(repo_owner='sacer11', repo_name='MLOps_movie_rec_apr25', mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("movie_user_model")

class UserKNNWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["knn_model"], "rb") as f:
            self.model = pickle.load(f)
    def predict(self, context, model_input):
        return self.model.kneighbors(model_input, return_distance=False).tolist()

def train_user_model(run_name="train_user_model"):
    print("🚀 Starte Training des user_model")
    data_dir = "/opt/airflow/data/processed"
    ratings_path = "/opt/airflow/data/raw/ratings.csv"
    matrix_path = f"{data_dir}/movie_matrix.csv"
    user_matrix_path = f"{data_dir}/user_matrix.csv"
    model_dir = "/opt/airflow/models"
    model_path = f"{model_dir}/user_model.pkl"

    print("📥 Lade Bewertungen und Movie-Embeddings")
    ratings = pd.read_csv(ratings_path)
    movie_embeddings = pd.read_csv(matrix_path, index_col=0)

    user_vectors, user_ids = [], []
    print("🔄 Erstelle User-Vektoren")
    for uid, group in ratings.groupby("userId"):
        rated = group[group["movieId"].isin(movie_embeddings.index)]
        if rated.empty:
            continue
        user_vector = movie_embeddings.loc[rated["movieId"]].mean(axis=0)
        user_vectors.append(user_vector)
        user_ids.append(uid)

    user_df = pd.DataFrame(user_vectors, index=user_ids)
    user_df.to_csv(user_matrix_path)
    print(f"✅ {len(user_df)} User-Vektoren gespeichert unter {user_matrix_path}")

    X = user_df.values
    print("📊 Trainiere NearestNeighbors-Modell")
    model = NearestNeighbors(n_neighbors=10, algorithm="ball_tree")
    model.fit(X)

    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Modell gespeichert unter {model_path}")

    print("📡 Starte MLflow-Tracking")
    with mlflow.start_run(run_name="train_user_model") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "train_user_model")
        mlflow.log_param("model_type", "user_knn")
        mlflow.log_param("n_neighbors", 10)
        mlflow.log_param("algorithm", "ball_tree")
        mlflow.log_metric("n_users", len(user_df))
        mlflow.log_artifact(user_matrix_path, artifact_path="features")
        # Modell-Backup, aber nicht im pyfunc-Verzeichnis!
        mlflow.log_artifact(model_path, artifact_path="backup_model")
        # Pyfunc-Modell für Registry
        signature = infer_signature(X, model.kneighbors(X[:5])[1])
        mlflow.pyfunc.log_model(
            artifact_path="user_knn_pyfunc",
            python_model=UserKNNWrapper(),
            artifacts={"knn_model": model_path},
            signature=signature,
            input_example=X[:5]
        )
    print("🏁 Training abgeschlossen und geloggt.")

if __name__ == "__main__":
    train_user_model()