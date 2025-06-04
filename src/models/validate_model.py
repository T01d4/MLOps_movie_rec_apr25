# src/models/validate_model.py

import pickle
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
import os
import logging
import argparse
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

MODEL_DIR = "/opt/airflow/models"

# Registry-Namen
MODEL_REGISTRY_NAMES = {
    "hybrid": "hybrid_model",
    "user": "user_model",
    "hybrid_dl": "hybrid_deep_model",
    "user_dl": "user_deep_model"
}
MODEL_FILE_PATHS = {
    "hybrid": os.path.join(MODEL_DIR, "hybrid_model.pkl"),
    "user": os.path.join(MODEL_DIR, "user_model.pkl"),
    "hybrid_dl": os.path.join(MODEL_DIR, "hybrid_deep_knn.pkl"),
    "user_dl": os.path.join(MODEL_DIR, "user_deep_knn.pkl")
}

def load_matrix_with_features(matrix_path, features_path=None, index_col=0):
    if features_path:
        with open(features_path, "r") as f:
            features = [line.strip() for line in f.readlines()]
        features_no_index = features[1:] if features[0].lower() in ["movieid", "userid"] else features
        df = pd.read_csv(matrix_path, index_col=index_col)
        assert list(df.columns) == features_no_index, (
            f"Spalten stimmen nicht! Matrix: {len(df.columns)} Features: {len(features_no_index)}"
        )
        return df, features_no_index
    else:
        # Deep-Learning-Matrix ohne Features
        df = pd.read_csv(matrix_path, index_col=index_col)
        return df, df.columns.tolist()

def validate_models(pipeline_type="classic", test_user_count=100):
    logging.info(f"🚀 Starte Validierung ({pipeline_type.upper()})")

    try:
        ratings = pd.read_csv("/opt/airflow/data/raw/ratings.csv")
        if pipeline_type == "classic":
            hybrid_matrix, hybrid_features = load_matrix_with_features(
                "/opt/airflow/data/processed/hybrid_matrix.csv",
                "/opt/airflow/data/processed/hybrid_matrix_features.txt"
            )
            user_matrix, user_features = load_matrix_with_features(
                "/opt/airflow/data/processed/user_matrix.csv",
                "/opt/airflow/data/processed/user_matrix_features.txt"
            )
            hybrid_model = pickle.load(open(MODEL_FILE_PATHS["hybrid"], "rb"))
            user_model = pickle.load(open(MODEL_FILE_PATHS["user"], "rb"))
        else:
            hybrid_matrix, hybrid_features = load_matrix_with_features(
                "/opt/airflow/data/processed/hybrid_deep_embedding.csv"
            )
            user_matrix, user_features = load_matrix_with_features(
                "/opt/airflow/data/processed/user_deep_embedding.csv"
            )
            hybrid_model = pickle.load(open(MODEL_FILE_PATHS["hybrid_dl"], "rb"))
            user_model = pickle.load(open(MODEL_FILE_PATHS["user_dl"], "rb"))
        logging.info("📥 Daten & Modelle geladen – Beginne Evaluation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Laden der Daten/Modelle: {e}", exc_info=True)
        return

    test_users = list(user_matrix.index)[:test_user_count]
    hybrid_scores, user_scores, valid_users = [], [], []

    for uid in test_users:
        try:
            uvec_user = user_matrix.loc[uid].values.reshape(1, -1)
            # Für klassisch: Movie-Features, für DL: Embeddings mit Index uid
            if pipeline_type == "classic":
                uvec_hybrid = hybrid_matrix.loc[uid].values.reshape(1, -1)
            else:
                uvec_hybrid = hybrid_matrix.loc[uid].values.reshape(1, -1)
            # Check, ob Feature-Anzahl stimmt
            if uvec_user.shape[1] != user_model.n_features_in_:
                raise ValueError(f"user_model erwartet {user_model.n_features_in_} Features, hat aber {uvec_user.shape[1]}")
            if uvec_hybrid.shape[1] != hybrid_model.n_features_in_:
                raise ValueError(f"hybrid_model erwartet {hybrid_model.n_features_in_} Features, hat aber {uvec_hybrid.shape[1]}")
            _, idxs_hybrid = hybrid_model.kneighbors(uvec_hybrid)
            rec_hybrid = hybrid_matrix.index[idxs_hybrid[0]]
            hit_hybrid = ratings[(ratings["userId"] == uid) & (ratings["movieId"].isin(rec_hybrid))]
            hybrid_scores.append(1 if not hit_hybrid.empty else 0)
            _, idxs_user = user_model.kneighbors(uvec_user)
            rec_user = user_matrix.index[idxs_user[0]]
            hit_user = ratings[(ratings["userId"] == uid) & (ratings["movieId"].isin(rec_user))]
            user_scores.append(1 if not hit_user.empty else 0)
            valid_users.append(uid)
        except Exception as e:
            logging.warning(f"⚠️ Fehler bei User {uid}: {e}")
            continue

    if not valid_users:
        logging.error("❌ Keine gültigen Nutzer zur Auswertung!")
        return

    hybrid_mean = float(np.mean(hybrid_scores))
    user_mean = float(np.mean(user_scores))
    logging.info(f"📊 precision_10_hybrid: {hybrid_mean:.2f}")
    logging.info(f"📊 precision_10_user:   {user_mean:.2f}")

    try:
        with mlflow.start_run(run_name=f"validate_predictions_{pipeline_type}") as run:
            mlflow.set_tag("pipeline_type", pipeline_type)
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "validate_models")
            mlflow.log_param("n_test_users", len(valid_users))
            mlflow.log_metric(f"precision_10_hybrid_{pipeline_type}", hybrid_mean)
            mlflow.log_metric(f"precision_10_user_{pipeline_type}", user_mean)
            score_df = pd.DataFrame({
                "user_id": valid_users,
                "hybrid_score": hybrid_scores,
                "user_score": user_scores
            })
            score_path = f"/opt/airflow/data/processed/validation_scores_{pipeline_type}.csv"
            score_df.to_csv(score_path, index=False)
            mlflow.log_artifact(score_path, artifact_path="validation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Logging in MLflow: {e}", exc_info=True)
        return

    logging.info("🎉 Validation abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_type", type=str, default="classic", choices=["classic", "dl"])
    parser.add_argument("--test_user_count", type=int, default=100)
    args = parser.parse_args()
    validate_models(pipeline_type=args.pipeline_type, test_user_count=args.test_user_count)