# src/models/train_user_deep_model.py

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

class UserAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(UserAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
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

def train_autoencoder(user_df, latent_dim=32, epochs=25, lr=1e-3, batch_size=64):
    X = user_df.values.astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UserAutoEncoder(X.shape[1], latent_dim).to(device)
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
    embedding_df = pd.DataFrame(embeddings, index=user_df.index)
    return model, embedding_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=32)
    args = parser.parse_args()
    n_neighbors = args.n_neighbors
    latent_dim = args.latent_dim
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("user_deep_model_exp")  # <--- KLARER EXPERIMENT-NAME!

    data_dir = "/opt/airflow/data/processed"
    ratings_path = f"{data_dir}/../raw/ratings.csv"
    movie_matrix_path = f"{data_dir}/movie_matrix.csv"

    ratings = pd.read_csv(ratings_path)
    movie_embeddings = pd.read_csv(movie_matrix_path, index_col=0).sort_index()
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
    embedding_path = f"{data_dir}/user_deep_embedding.csv"

    model, embedding_df = train_autoencoder(user_df, latent_dim=latent_dim, epochs=25)
    embedding_df.to_csv(embedding_path)
    print(f"✅ User-Embeddings gespeichert unter {embedding_path}")

    deep_knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    deep_knn.fit(embedding_df.values)
    model_dir = "/opt/airflow/models"
    os.makedirs(model_dir, exist_ok=True)
    deep_knn_path = f"{model_dir}/user_deep_knn.pkl"
    with open(deep_knn_path, "wb") as f:
        pickle.dump(deep_knn, f)
    print(f"✅ Deep KNN Modell gespeichert unter {deep_knn_path}")

    with mlflow.start_run(run_name="train_user_deep_model") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "train_user_deep_model")
        mlflow.set_tag("model_group", "deep_learning")
        mlflow.log_param("model_type", "user_deep_knn")
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("algorithm", "ball_tree")
        mlflow.log_metric("n_users", len(embedding_df))
        mlflow.log_artifact(embedding_path, artifact_path="features")
        mlflow.log_artifact(deep_knn_path, artifact_path="backup_model")
        embedding_feature_names = [f"emb_{i}" for i in range(embedding_df.shape[1])]
        embedding_df.columns = embedding_feature_names
        signature = infer_signature(
            embedding_df,
            deep_knn.kneighbors(embedding_df.values[:5])[1]
        )
        mlflow.pyfunc.log_model(
            artifact_path="user_deep_knn_pyfunc",
            python_model=KNNPyFuncWrapper(),
            artifacts={"knn_model": deep_knn_path},
            signature=signature,
            input_example=embedding_df.iloc[:5],
            registered_model_name="user_deep_model"  # <--- KLARE REGISTRY!
        )
        client = MlflowClient()
        model_name = "user_deep_model"
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
            client.set_model_version_tag(model_name, model_version, "algorithm", "ball_tree")
            print(f"📝 Tags für Modellversion {model_version} gesetzt: n_neighbors={n_neighbors}, latent_dim={latent_dim}")
        else:
            print("⚠️ Konnte Modellversion für Tagging nicht bestimmen.")

    print("🏁 Deep User-Model Training abgeschlossen und geloggt.")

if __name__ == "__main__":
    main()