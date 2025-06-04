# src/models/train_hybrid_deep_model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import argparse
from mlflow.tracking import MlflowClient

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

def train_autoencoder(X, latent_dim=64, epochs=30, lr=1e-3, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridAutoEncoder(X.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=batch_size, shuffle=True)
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
        print(f"Epoch {epoch+1}/{epochs}: Loss = {np.mean(losses):.4f}")
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(torch.from_numpy(X).to(device)).cpu().numpy()
    return model, embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=64)
    args = parser.parse_args()
    n_neighbors = args.n_neighbors
    latent_dim = args.latent_dim

    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("hybrid_deep_model_exp")  # <--- KLARER EXPERIMENT-NAME!

    matrix_path = "/opt/airflow/data/processed/hybrid_matrix.csv"
    features_path = "/opt/airflow/data/processed/hybrid_matrix_features.txt"
    df = pd.read_csv(matrix_path)
    with open(features_path, "r") as f:
        features = [line.strip() for line in f.readlines()]
    X = df.drop(columns=["movieId"]).values.astype(np.float32)
    movie_ids = df["movieId"].values

    model, hybrid_embeddings = train_autoencoder(X, latent_dim=latent_dim, epochs=30)
    embedding_df = pd.DataFrame(hybrid_embeddings, index=movie_ids)
    embedding_path = "/opt/airflow/data/processed/hybrid_deep_embedding.csv"
    embedding_df.to_csv(embedding_path)
    print(f"✅ Hybrid-Embeddings gespeichert unter {embedding_path}")

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(hybrid_embeddings)
    model_dir = "/opt/airflow/models"
    os.makedirs(model_dir, exist_ok=True)
    knn_path = os.path.join(model_dir, "hybrid_deep_knn.pkl")
    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    print(f"✅ Deep KNN Modell gespeichert unter {knn_path}")

    with mlflow.start_run(run_name="train_hybrid_deep_model") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "train_hybrid_deep_model")
        mlflow.log_param("model_type", "hybrid_deep_knn")
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("n_movies", len(embedding_df))
        mlflow.log_artifact(embedding_path, artifact_path="features")
        mlflow.log_artifact(knn_path, artifact_path="backup_model")
        embedding_feature_names = [f"emb_{i}" for i in range(hybrid_embeddings.shape[1])]
        signature = infer_signature(
            pd.DataFrame(hybrid_embeddings, columns=embedding_feature_names),
            knn.kneighbors(hybrid_embeddings[:5])[1]
        )
        mlflow.pyfunc.log_model(
            artifact_path="hybrid_deep_knn_pyfunc",
            python_model=KNNPyFuncWrapper(),
            artifacts={"knn_model": knn_path},
            signature=signature,
            input_example=pd.DataFrame(hybrid_embeddings[:5], columns=embedding_feature_names),
            registered_model_name="hybrid_deep_model"  # <--- KLARE REGISTRY!
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
            print(f"📝 Tags für Modellversion {model_version} gesetzt: n_neighbors={n_neighbors}, latent_dim={latent_dim}")
        else:
            print("⚠️ Konnte Modellversion für Tagging nicht bestimmen.")

    print("🏁 Deep Hybrid-Model Training abgeschlossen und geloggt.")

if __name__ == "__main__":
    main()