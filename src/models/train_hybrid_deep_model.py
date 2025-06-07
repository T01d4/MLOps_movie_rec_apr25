# src/models/train_hybrid_deep_model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import argparse
from mlflow.tracking import MlflowClient
import logging

class HybridAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(HybridAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_embedding(self, x):
        return self.encoder(x)

class KNNPyFuncWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        knn_path = context.artifacts["knn_model"]
        with open(knn_path, "rb") as f:
            self.knn = pickle.load(f)

    def predict(self, context, model_input):
        return self.knn.kneighbors(model_input.values)[1]

def train_hybrid_deep_model(n_neighbors=10, latent_dim=64, epochs=30, tfidf_features=300, save_matrix_csv=True):
    logger = logging.getLogger(__name__)
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("hybrid_deep_model")

    # --- Daten laden & Feature-Matrix erzeugen ---
    logger.info("📥 Lade Rohdaten und baue Hybrid-Feature-Matrix ...")
    movies = pd.read_csv("/opt/airflow/data/raw/movies.csv")
    tags = pd.read_csv("/opt/airflow/data/raw/tags.csv")
    scores = pd.read_csv("/opt/airflow/data/raw/genome-scores.csv")

    tags = tags.dropna(subset=["tag"])
    tags_combined = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(str(x) for x in t)).reset_index()
    movies = pd.merge(movies, tags_combined, on="movieId", how="left")
    movies["combined"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"].fillna("")

    vectorizer = TfidfVectorizer(max_features=tfidf_features)
    content_embeddings = vectorizer.fit_transform(movies["combined"])
    collab_embeddings = scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

    common_ids = movies[movies["movieId"].isin(collab_embeddings.index)].copy()
    content_embeddings = content_embeddings[[i for i, mid in enumerate(movies["movieId"]) if mid in common_ids["movieId"].values]]
    collab_embeddings = collab_embeddings.loc[common_ids["movieId"]]

    scaler = MinMaxScaler()
    collab_scaled = scaler.fit_transform(collab_embeddings)
    content_scaled = scaler.fit_transform(content_embeddings.toarray())
    hybrid_matrix = np.hstack([collab_scaled, content_scaled])

    # Feature-Namen erzeugen (Kollaborativ + Content)
    collab_feature_names = [f"collab_{col}" for col in collab_embeddings.columns]
    content_feature_names = [f"tfidf_{i}" for i in range(content_scaled.shape[1])]
    feature_names = ["movieId"] + collab_feature_names + content_feature_names

    # DataFrame: movieId als explizite SPALTE!
    hybrid_df = pd.DataFrame(
        hybrid_matrix,
        columns=collab_feature_names + content_feature_names
    )
    hybrid_df.insert(0, "movieId", common_ids["movieId"].values)

    logger.info(f"📐 Hybrid-Matrix erstellt mit Shape: {hybrid_df.shape}, Feature-Namen: {len(feature_names)}")

    # Optional: Matrix speichern (mit Spaltenheader)
    if save_matrix_csv:
        matrix_path = "/opt/airflow/data/processed/hybrid_matrix.csv"
        hybrid_df.to_csv(matrix_path, index=False)
        logger.info(f"💾 Hybrid-Matrix gespeichert unter {matrix_path}")

    # Für Autoencoder: ohne movieId!
    X = hybrid_df.drop(columns=["movieId"]).values.astype(np.float32)
    movie_ids = hybrid_df["movieId"].values

    # --- Autoencoder Training ---
    logger.info("🚀 Starte Training Autoencoder ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridAutoEncoder(X.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=128, shuffle=True)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {np.mean(losses):.4f}")
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(torch.from_numpy(X).to(device)).cpu().numpy()

    embedding_feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, index=movie_ids, columns=embedding_feature_names)
    embedding_path = "/opt/airflow/data/processed/hybrid_deep_embedding.csv"
    embedding_df.to_csv(embedding_path)
    logger.info(f"✅ Hybrid-Embeddings gespeichert unter {embedding_path}")
    logger.info(f"✅ Hybrid-Embeddings gespeichert unter {embedding_path}")

    # --- KNN Training ---
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)
    model_dir = "/opt/airflow/models"
    os.makedirs(model_dir, exist_ok=True)
    knn_path = os.path.join(model_dir, "hybrid_deep_knn.pkl")
    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    logger.info(f"✅ Deep KNN Modell gespeichert unter {knn_path}")

    # --- MLflow Logging & Metadaten ---
    with mlflow.start_run(run_name="train_hybrid_deep_model") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "train_hybrid_deep_model")
        mlflow.set_tag("model_group", "deep_learning")
        mlflow.log_param("model_type", "hybrid_deep_knn")
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("tfidf_features", tfidf_features)
        mlflow.log_param("algorithm", "cosine")
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("n_movies", len(embedding_df))
        mlflow.log_artifact(embedding_path, artifact_path="features")
        mlflow.log_artifact(knn_path, artifact_path="backup_model")

        embedding_feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
        signature = infer_signature(
            pd.DataFrame(embeddings, columns=embedding_feature_names),
            knn.kneighbors(embeddings[:5])[1]
        )
        mlflow.pyfunc.log_model(
            artifact_path="hybrid_deep_knn_pyfunc",
            python_model=KNNPyFuncWrapper(),
            artifacts={"knn_model": knn_path},
            signature=signature,
            input_example=pd.DataFrame(embeddings[:5], columns=embedding_feature_names),
            registered_model_name="hybrid_deep_model"
        )
        client = MlflowClient()
        model_name = "hybrid_deep_model"
        run_id = run.info.run_id
        model_version = None
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.run_id == run_id:
                model_version = v.version
                break
        if model_version:
            client.set_model_version_tag(model_name, model_version, "n_neighbors", str(n_neighbors))
            client.set_model_version_tag(model_name, model_version, "latent_dim", str(latent_dim))
            client.set_model_version_tag(model_name, model_version, "tfidf_features", str(tfidf_features))
            client.set_model_version_tag(model_name, model_version, "algorithm", "cosine")
            client.set_model_version_tag(model_name, model_version, "precision_10", 0.0)
            logger.info(f"📝 Tags für Modellversion {model_version} gesetzt: n_neighbors={n_neighbors}, latent_dim={latent_dim}")
        else:
            logger.warning("⚠️ Konnte Modellversion für Tagging nicht bestimmen.")

    logger.info("🏁 Deep Hybrid-Model Training abgeschlossen und geloggt.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tfidf_features", type=int, default=300)
    args = parser.parse_args()
    train_hybrid_deep_model(
        n_neighbors=args.n_neighbors,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        tfidf_features=args.tfidf_features,
    )