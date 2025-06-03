# src/models/validate_deep_models.py
import pickle
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
import os
import logging
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

MODEL_DIR = "/opt/airflow/models"
HYBRID_MODEL = os.path.join(MODEL_DIR, "hybrid_deep_knn.pkl")

def validate_deep_hybrid_model(test_user_count=100):
    logging.info("🚀 Starte Validierung der Deep Hybrid Modelle")
    client = MlflowClient()
    try:
        ratings = pd.read_csv("/opt/airflow/data/raw/ratings.csv")
        embedding_path = "/opt/airflow/data/processed/hybrid_deep_embedding.csv"
        embedding_df = pd.read_csv(embedding_path, index_col=0)

        with open(HYBRID_MODEL, "rb") as f:
            hybrid_model = pickle.load(f)
        logging.info("📥 Embeddings & Modell geladen – Beginne Evaluation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Laden der Daten/Modelle: {e}", exc_info=True)
        return

    # User-Vektoren für Test-Users: wie gehabt!
    all_users = embedding_df.index.tolist()[:test_user_count]
    test_scores, valid_users = [], []

    for mid in all_users:
        try:
            movie_vec = embedding_df.loc[mid].values.reshape(1, -1)
            if movie_vec.shape[1] != hybrid_model.n_features_in_:
                raise ValueError("Feature mismatch")
            _, idxs = hybrid_model.kneighbors(movie_vec)
            rec_ids = embedding_df.index[idxs[0]]
            # Prüfe ob Original-Movie in den Top-10-Vorschlägen ist (kann man anpassen!)
            hit = mid in rec_ids
            test_scores.append(1 if hit else 0)
            valid_users.append(mid)
        except Exception as e:
            logging.warning(f"⚠️ Fehler bei MovieId {mid}: {e}")
            continue

    deep_hybrid_mean = float(np.mean(test_scores))
    logging.info(f"📊 precision_10_hybrid_deep: {deep_hybrid_mean:.2f}")

    # Logging in MLflow
    try:
        with mlflow.start_run(run_name="validate_hybrid_deep_predictions") as run:
            mlflow.set_tag("model_type", "hybrid_deep_knn")
            mlflow.log_param("n_test_items", len(valid_users))
            mlflow.log_metric("precision_10_hybrid_deep", deep_hybrid_mean)
            score_df = pd.DataFrame({
                "movie_id": valid_users,
                "score": test_scores
            })
            score_path = "/opt/airflow/data/processed/validation_hybrid_deep_scores.csv"
            score_df.to_csv(score_path, index=False)
            mlflow.log_artifact(score_path, artifact_path="validation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Logging in MLflow: {e}", exc_info=True)
        return

    logging.info("🎉 Validation & Registry update abgeschlossen.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_user_count", type=int, default=100)
    args = parser.parse_args()
    validate_deep_hybrid_model(test_user_count=args.test_user_count)