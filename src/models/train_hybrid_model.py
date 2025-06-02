# src/models/train_hybrid_model.py

import mlflow
import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import argparse

class HybridKNNWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["knn_model"], "rb") as f:
            self.knn = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> list:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame!")
        if model_input.shape[1] != self.knn.n_features_in_:
            raise ValueError(
                f"Feature mismatch: Model expects {self.knn.n_features_in_} features, "
                f"but input has {model_input.shape[1]}"
            )
        _, indices = self.knn.kneighbors(model_input.values)
        return indices.tolist()

def train_hybrid_model(n_neighbors=10, tfidf_features=300):
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starte Hybrid-Modelltraining")
    try:
        # === .env laden und MLflow setzen ===
        load_dotenv()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("movie_hybrid_model")

        # === Daten laden ===
        movies = pd.read_csv("/opt/airflow/data/raw/movies.csv")
        tags = pd.read_csv("/opt/airflow/data/raw/tags.csv")
        scores = pd.read_csv("/opt/airflow/data/raw/genome-scores.csv")
        logger.info("📥 Rohdaten erfolgreich geladen")

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

        # === DataFrame: movieId als explizite SPALTE! ===
        hybrid_df = pd.DataFrame(
            hybrid_matrix,
            columns=collab_feature_names + content_feature_names
        )
        hybrid_df.insert(0, "movieId", common_ids["movieId"].values)

        logger.info(f"📐 Hybrid-Matrix erstellt mit Shape: {hybrid_df.shape}, Feature-Namen: {len(feature_names)}")
        assert hybrid_df.shape[1] == len(feature_names), \
            f"Mismatch: Matrix has {hybrid_df.shape[1]} features, but names: {len(feature_names)}"

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(hybrid_df.drop(columns=["movieId"]).values)
        logger.info("🤖 KNN-Modell trainiert")

        model_dir = "/opt/airflow/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hybrid_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(knn, f)

        # Speichere Matrix **mit movieId als explizite Spalte**
        matrix_path = "/opt/airflow/data/processed/hybrid_matrix.csv"
        hybrid_df.to_csv(matrix_path, index=False)

        features_path = "/opt/airflow/data/processed/hybrid_matrix_features.txt"
        with open(features_path, "w") as f:
            f.write("\n".join(feature_names))
        logger.info(f"✅ Feature-Schema gespeichert unter {features_path}")

        with mlflow.start_run(run_name="train_hybrid_model") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "train_hybrid_model")
            mlflow.log_param("model_type", "hybrid_knn")
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("vectorizer_max_features", tfidf_features)
            mlflow.log_metric("n_samples", hybrid_df.shape[0])
            mlflow.log_metric("n_features", hybrid_df.shape[1])
            signature = infer_signature(
                hybrid_df.drop(columns=["movieId"]),
                knn.kneighbors(hybrid_df.drop(columns=["movieId"]).values)[1]
            )
            # Modell registrieren
            mlflow.pyfunc.log_model(
                artifact_path="hybrid_knn_pyfunc",
                python_model=HybridKNNWrapper(),
                artifacts={"knn_model": model_path},
                signature=signature,
                input_example=hybrid_df.drop(columns=["movieId"]).iloc[:2],
                registered_model_name="movie_model"
            )
            mlflow.log_artifact(matrix_path, artifact_path="features")
            mlflow.log_artifact(features_path, artifact_path="features")

            # ============ NEU: Modellversion als Tag setzen ===============
            client = MlflowClient()
            model_name = "movie_model"
            run_id = run.info.run_id
            # Suche die Modellversion für diesen Run
            model_version = None
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                if v.run_id == run_id:
                    model_version = v.version
                    break
            if model_version:
                client.set_model_version_tag(model_name, model_version, "n_neighbors", str(n_neighbors))
                client.set_model_version_tag(model_name, model_version, "tfidf_features", str(tfidf_features))
                logger.info(f"📝 Tags für Modellversion {model_version} gesetzt: n_neighbors={n_neighbors}, tfidf_features={tfidf_features}")
            else:
                logger.warning("⚠️ Konnte Modellversion für Tagging nicht bestimmen.")

        logger.info("✅ Modell & Matrix erfolgreich gespeichert und in MLflow geloggt")
    except Exception as e:
        logger.error(f"❌ Fehler im Hybrid-Training: {e}", exc_info=True)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--tfidf_features", type=int, default=300)
    args = parser.parse_args()
    train_hybrid_model(n_neighbors=args.n_neighbors, tfidf_features=args.tfidf_features)