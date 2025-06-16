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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import argparse
from mlflow.tracking import MlflowClient
import logging

# === Prometheus Monitoring (Loss pro Epoche + Drift Detection) ===
from prometheus_client import Gauge, CollectorRegistry, write_to_textfile
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

registry = CollectorRegistry()
LOSS_GAUGE = Gauge("training_loss", "Training loss per epoch", ["epoch"], registry=registry)
EPOCH_GAUGE = Gauge("current_epoch", "Current epoch", registry=registry)



DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models")
REPORT_DIR= os.getenv("REPORT_DIR", "/opt/airflow/reports")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


class HybridAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super(HybridAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
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

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        return self.knn.kneighbors(model_input.values)[1]

def train_hybrid_deep_model(save_matrix_csv=True):
    CONFIG_PATH = os.path.join(PROCESSED_DIR, "pipeline_conf.json")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    n_neighbors = config["n_neighbors"]
    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]
    tfidf_features = config["tfidf_features"]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    metric = config["metric"]
    content_weight = config["content_weight"]
    collab_weight = config["collab_weight"]
    power_factor = config["power_factor"]
    drop_threshold = config.get("drop_threshold", None)

    logger = logging.getLogger(__name__)
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("hybrid_deep_model")

    logger.info("📥 Loading raw data and building hybrid feature matrix ...")
    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))
    scores = pd.read_csv(os.path.join(RAW_DIR, "genome-scores.csv"))

    tags = tags.dropna(subset=["tag"])
    tags_combined = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(str(x) for x in t)).reset_index()
    movies = pd.merge(movies, tags_combined, on="movieId", how="left")
    movies["combined"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"].fillna("")

    vectorizer = TfidfVectorizer(max_features=tfidf_features)
    content_embeddings = vectorizer.fit_transform(movies["combined"])
    collab_embeddings = scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

    if drop_threshold is not None:
        tag_means = collab_embeddings.mean()
        collab_embeddings = collab_embeddings.loc[:, tag_means >= drop_threshold]

    common_ids = movies[movies["movieId"].isin(collab_embeddings.index)].copy()
    content_embeddings = content_embeddings[[i for i, mid in enumerate(movies["movieId"]) if mid in common_ids["movieId"].values]]
    collab_embeddings = collab_embeddings.loc[common_ids["movieId"]]

    scaler = MinMaxScaler()
    collab_scaled = scaler.fit_transform(collab_embeddings)
    content_scaled = scaler.fit_transform(content_embeddings.toarray())

    content_scaled *= content_weight
    collab_scaled *= collab_weight

    if power_factor != 1.0:
        content_scaled = np.power(content_scaled, power_factor)
        collab_scaled = np.power(collab_scaled, power_factor)

    hybrid_matrix = np.hstack([collab_scaled, content_scaled])
    collab_feature_names = [f"collab_{col}" for col in collab_embeddings.columns]
    content_feature_names = [f"tfidf_{i}" for i in range(content_scaled.shape[1])]
    feature_names = ["movieId"] + collab_feature_names + content_feature_names

    hybrid_df = pd.DataFrame(hybrid_matrix, columns=collab_feature_names + content_feature_names)
    hybrid_df.insert(0, "movieId", common_ids["movieId"].values)

    logger.info(f"📐 Hybrid matrix created with shape: {hybrid_df.shape}, number of features: {len(feature_names)}")

    if save_matrix_csv:
        matrix_path = os.path.join(PROCESSED_DIR, "hybrid_matrix.csv")
        hybrid_df.to_csv(matrix_path, index=False)
        logger.info(f"💾 Hybrid matrix saved at {matrix_path}")

    X = hybrid_df.drop(columns=["movieId"]).values.astype(np.float32)
    movie_ids = hybrid_df["movieId"].values

    logger.info("🚀 Starting Autoencoder training ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridAutoEncoder(X.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=batch_size, shuffle=True)

    losses = []
    early_stop_patience = 8
    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}: Loss = {mean_loss:.4f}")

        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                logger.info(f"⛔️ EarlyStopping nach {epoch+1} Epochen (kein Fortschritt seit {early_stop_patience})")
                break

    # ✅ Prometheus nach dem Training beschreiben
    for epoch_idx, epoch_loss in enumerate(losses):
        LOSS_GAUGE.labels(epoch=str(epoch_idx + 1)).set(epoch_loss)
        EPOCH_GAUGE.set(epoch_idx + 1)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(torch.from_numpy(X).to(device)).cpu().numpy()

    embedding_feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, index=movie_ids, columns=embedding_feature_names)
    embedding_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
    embedding_df.to_csv(embedding_path)
    logger.info(f"✅ Hybrid embeddings saved at {embedding_path}")

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(embeddings)
    os.makedirs(MODEL_DIR, exist_ok=True)
    knn_path = os.path.join(MODEL_DIR, "hybrid_deep_knn.pkl")
    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    logger.info(f"✅ Deep KNN model saved at {knn_path}")

    with mlflow.start_run(run_name="train_hybrid_deep_model") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "train_hybrid_deep_model")
        mlflow.set_tag("model_group", "deep_learning")

        mlflow.log_param("model_type", "hybrid_deep_knn")
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("tfidf_features", tfidf_features)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("metric", metric)
        mlflow.log_param("content_weight", content_weight)
        mlflow.log_param("collab_weight", collab_weight)
        mlflow.log_param("power_factor", power_factor)
        if drop_threshold is not None:
            mlflow.log_param("drop_threshold", drop_threshold)

        mlflow.log_metric("n_movies", len(embedding_df))
        mlflow.log_artifact(embedding_path, artifact_path="features")
        mlflow.log_artifact(knn_path, artifact_path="backup_model")
        mlflow.log_artifact(CONFIG_PATH, artifact_path="config")
        
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
            client.set_model_version_tag(model_name, model_version, "hidden_dim", str(hidden_dim))
            client.set_model_version_tag(model_name, model_version, "tfidf_features", str(tfidf_features))
            client.set_model_version_tag(model_name, model_version, "epochs", str(epochs))
            client.set_model_version_tag(model_name, model_version, "lr", str(lr))
            client.set_model_version_tag(model_name, model_version, "batch_size", str(batch_size))
            client.set_model_version_tag(model_name, model_version, "metric", metric)
            client.set_model_version_tag(model_name, model_version, "content_weight", str(content_weight))
            client.set_model_version_tag(model_name, model_version, "collab_weight", str(collab_weight))
            client.set_model_version_tag(model_name, model_version, "power_factor", str(power_factor))
            if drop_threshold is not None:
                client.set_model_version_tag(model_name, model_version, "drop_threshold", str(drop_threshold))
            client.set_model_version_tag(model_name, model_version, "precision_10", 0.0)

            logger.info(
                f"📝 Tags set for model version {model_version}: "
                f"n_neighbors={n_neighbors}, latent_dim={latent_dim}, hidden_dim={hidden_dim}, "
                f"tfidf_features={tfidf_features}, epochs={epochs}, lr={lr}, batch_size={batch_size}, "
                f"metric={metric}, content_weight={content_weight}, collab_weight={collab_weight}, "
                f"power_factor={power_factor}, precision_10=0.0"
            )
        else:
            logger.warning("⚠️ Could not determine model version for tagging.")

    logger.info("🏁 Deep hybrid model training completed and logged.")

    # === Drift Detection & Prometheus Monitoring ===
    try:
        ref_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding_best.csv")
        if os.path.exists(ref_path):
            reference_df = pd.read_csv(ref_path)
            current_df = embedding_df.reset_index()

            # 👉 Gemeinsame Spalten auswählen & sortieren
            common_cols = sorted(set(reference_df.columns) & set(current_df.columns))
            if not common_cols:
                raise ValueError("⚠️ Keine gemeinsamen Spalten für Drift-Analyse gefunden.")

            reference_df = reference_df[common_cols]
            current_df = current_df[common_cols]

            # 🧠 ColumnMapping dynamisch erstellen
            column_mapping = ColumnMapping(numerical_features=common_cols)

            # 🧪 Report ausführen
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
            drift_result = drift_report.as_dict()

            # 🎯 Driftwert extrahieren (robust)
            drift_score = None
            for m in drift_result.get("metrics", []):
                if m.get("metric") == "DatasetDriftMetric":
                    drift_score = m.get("result", {}).get("drift_share", None)
                    break

            drift_alert = int(drift_score > 0.2) if drift_score is not None else 0

            # 🔧 Prometheus Metriken setzen
            DRIFT_GAUGE = Gauge("model_drift_alert", "Drift detected after training", registry=registry)
            DRIFT_GAUGE.set(drift_alert)

            DRIFT_ALERT = Gauge("drift_alert", "Drift alert flag", ["model"], registry=registry)
            DRIFT_ALERT.labels(model="hybrid_deep_model").set(drift_alert)

            # 📄 Logging
            if drift_score is not None:
                logger.info(f"🔍 Drift detected in {drift_score:.2%} of features (Alert={drift_alert})")
            else:
                logger.warning("⚠️ Drift share could not be computed – value is None")
                logger.info(f"Ref shape: {reference_df.shape}, Cur shape: {current_df.shape}")

            # 💾 JSON speichern
            json_path = os.path.join(REPORT_DIR, "drift_metrics.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(drift_result, f, indent=2)
            logger.info(f"📄 Drift metrics saved to {json_path}")

        else:
            logger.warning(f"⚠️ Reference embedding for drift analysis not found at {ref_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to compute drift: {e}")

    # Prometheus-Datei schreiben
    try:
        prom_path = os.getenv("REPORT_DIR", "/app/reports")
        os.makedirs(prom_path, exist_ok=True)
        write_to_textfile(os.path.join(prom_path, "training_metrics.prom"), registry)
        logger.info("💾 Prometheus training metrics written to file.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to write Prometheus training metrics: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    train_hybrid_deep_model()