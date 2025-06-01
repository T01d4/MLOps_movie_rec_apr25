# src/models/train_hybrid_model.py

import mlflow
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import dagshub
# === .env laden und MLflow setzen ===
load_dotenv()
#dagshub.init(repo_owner='sacer11', repo_name='MLOps_movie_rec_apr25', mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("movie_hybrid_model")

class HybridKNNWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["knn_model"], "rb") as f:
            self.knn = pickle.load(f)
    def predict(self, context, model_input):
        _, indices = self.knn.kneighbors(model_input)
        return indices.tolist()

def train_hybrid_model():
    print("🚀 Starte Hybrid-Modelltraining")
    try:
        # === Daten laden ===
        movies = pd.read_csv("/opt/airflow/data/raw/movies.csv")
        tags = pd.read_csv("/opt/airflow/data/raw/tags.csv")
        scores = pd.read_csv("/opt/airflow/data/raw/genome-scores.csv")
        print("📥 Rohdaten erfolgreich geladen")

        tags = tags.dropna(subset=["tag"])
        tags_combined = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(str(x) for x in t)).reset_index()
        movies = pd.merge(movies, tags_combined, on="movieId", how="left")
        movies["combined"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"].fillna("")

        vectorizer = TfidfVectorizer(max_features=300)
        content_embeddings = vectorizer.fit_transform(movies["combined"])
        collab_embeddings = scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

        common_ids = movies[movies["movieId"].isin(collab_embeddings.index)].copy()
        content_embeddings = content_embeddings[[i for i, mid in enumerate(movies["movieId"]) if mid in common_ids["movieId"].values]]
        collab_embeddings = collab_embeddings.loc[common_ids["movieId"]]

        scaler = MinMaxScaler()
        collab_scaled = scaler.fit_transform(collab_embeddings)
        content_scaled = scaler.fit_transform(content_embeddings.toarray())
        hybrid_matrix = np.hstack([collab_scaled, content_scaled])
        print(f"📐 Hybrid-Matrix erstellt mit Shape: {hybrid_matrix.shape}")

        knn = NearestNeighbors(n_neighbors=10, metric="cosine")
        knn.fit(hybrid_matrix)
        print("🤖 KNN-Modell trainiert")

        model_dir = "/opt/airflow/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hybrid_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(knn, f)

        matrix_path = "/opt/airflow/data/processed/hybrid_matrix.csv"
        pd.DataFrame(hybrid_matrix).to_csv(matrix_path, index=False)

        with mlflow.start_run(run_name="train_hybrid_model") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "train_hybrid_model")
            mlflow.log_param("model_type", "hybrid_knn")
            mlflow.log_param("n_neighbors", 10)
            mlflow.log_param("vectorizer_max_features", 300)
            mlflow.log_metric("n_samples", hybrid_matrix.shape[0])
            mlflow.log_metric("n_features", hybrid_matrix.shape[1])
            signature = infer_signature(hybrid_matrix, knn.kneighbors(hybrid_matrix)[1])
            mlflow.pyfunc.log_model(
                artifact_path="hybrid_knn_pyfunc",
                python_model=HybridKNNWrapper(),
                artifacts={"knn_model": model_path},
                signature=signature,
                input_example=hybrid_matrix[:2]
            )
        print("✅ Modell & Matrix erfolgreich gespeichert und in MLflow geloggt")
    except Exception as e:
        print(f"❌ Fehler im Hybrid-Training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_hybrid_model()