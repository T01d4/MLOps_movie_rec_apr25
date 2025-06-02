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
HYBRID_MODEL = os.path.join(MODEL_DIR, "hybrid_model.pkl")
USER_MODEL = os.path.join(MODEL_DIR, "user_model.pkl")

def load_matrix_with_features(matrix_path, features_path):
    with open(features_path, "r") as f:
        features = [line.strip() for line in f.readlines()]
    # movieId als Index
    df = pd.read_csv(matrix_path, index_col=0)
    features_no_index = features[1:] if features[0].lower() == "movieid" else features
    assert list(df.columns) == features_no_index, (
        f"Spalten stimmen nicht! Matrix: {len(df.columns)} Features: {len(features_no_index)}"
    )
    return df, features_no_index

def validate_models(test_user_count=100):
    logging.info("🚀 Starte Validierung der Modelle")
    client = MlflowClient()
    try:
        ratings = pd.read_csv("/opt/airflow/data/raw/ratings.csv")
        movie_matrix = pd.read_csv("/opt/airflow/data/processed/movie_matrix.csv", index_col=0)

        # Matrizen und Features laden
        hybrid_matrix, hybrid_features = load_matrix_with_features(
            "/opt/airflow/data/processed/hybrid_matrix.csv",
            "/opt/airflow/data/processed/hybrid_matrix_features.txt"
        )
        user_matrix, user_features = load_matrix_with_features(
            "/opt/airflow/data/processed/user_matrix.csv",
            "/opt/airflow/data/processed/user_matrix_features.txt"
        )

        with open(HYBRID_MODEL, "rb") as f:
            hybrid_model = pickle.load(f)
        with open(USER_MODEL, "rb") as f:
            user_model = pickle.load(f)
        logging.info("📥 Daten & Modelle geladen – Beginne Evaluation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Laden der Daten/Modelle: {e}", exc_info=True)
        return

    all_users = user_matrix.index.tolist()
    test_users = all_users[:test_user_count]
    logging.info(f"Test-User Auswahl: {test_users[:5]} ... ({len(test_users)} User)")

    hybrid_scores, user_scores, valid_users = [], [], []

    for uid in test_users:
        try:
            uvec_user = user_matrix.loc[uid, user_features].values.reshape(1, -1)
            uvec_hybrid = hybrid_matrix.loc[uid, hybrid_features].values.reshape(1, -1)
            if uvec_user.shape[1] != user_model.n_features_in_:
                raise ValueError(f"user_model erwartet {user_model.n_features_in_} Features, hat aber {uvec_user.shape[1]}")
            if uvec_hybrid.shape[1] != hybrid_model.n_features_in_:
                raise ValueError(f"hybrid_model erwartet {hybrid_model.n_features_in_} Features, hat aber {uvec_hybrid.shape[1]}")
            _, idxs_hybrid = hybrid_model.kneighbors(uvec_hybrid)
            hit_hybrid = ratings[(ratings["userId"] == uid) & (ratings["movieId"].isin(movie_matrix.index[idxs_hybrid[0]]))]
            hybrid_scores.append(1 if not hit_hybrid.empty else 0)
            _, idxs_user = user_model.kneighbors(uvec_user)
            hit_user = ratings[(ratings["userId"] == uid) & (ratings["movieId"].isin(user_matrix.index[idxs_user[0]]))]
            user_scores.append(1 if not hit_user.empty else 0)
            valid_users.append(uid)
        except Exception as e:
            logging.warning(f"⚠️ Fehler bei User {uid}: {e}")
            continue

    if len(valid_users) == 0:
        logging.error("❌ Keine gültigen Nutzer zur Auswertung!")
        return

    hybrid_mean = float(np.mean(hybrid_scores))
    user_mean = float(np.mean(user_scores))
    logging.info(f"📊 precision_10_hybrid: {hybrid_mean:.2f}")
    logging.info(f"📊 precision_10_user:   {user_mean:.2f}")

    # Registry: Bestes Modell suchen (via precision Tag)
    model_name = "movie_model"
    best_version, best_score = None, -1
    for v in client.search_model_versions(f"name='{model_name}'"):
        prec = None
        if "precision_10_hybrid" in v.tags:
            try:
                prec = float(str(v.tags["precision_10_hybrid"]))
            except Exception:
                continue
        if prec is not None and prec > best_score:
            best_score = prec
            best_version = v.version

    # Neuestes Modell suchen (höchste Versionnummer)
    versions = sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version))
    latest = versions[-1] if versions else None

    # --- Tagge die Validierungsparameter ---
    if latest:
        client.set_model_version_tag(model_name, latest.version, "validation_user_count", str(test_user_count))
        logging.info(f"📝 Tag für Modellversion {latest.version} gesetzt: validation_user_count={test_user_count}")

    # Precision für das "zu validierende" (neueste) Modell nehmen
    new_better = False
    if latest and hybrid_mean > best_score:
        new_better = True
        client.set_model_version_tag(model_name, latest.version, "precision_10_hybrid", str(hybrid_mean))
        client.set_registered_model_alias(model_name, "best_model", latest.version)
        logging.info(f"🎉 NEUES BESTES MODELL: Version {latest.version} mit precision_10_hybrid {hybrid_mean} – Alias @best_model gesetzt.")
    else:
        if latest:
            client.set_model_version_tag(model_name, latest.version, "precision_10_hybrid", str(hybrid_mean))
        logging.info(f"⚠️ Modell war nicht besser als das aktuelle @best_model (Score: {hybrid_mean} vs. {best_score}) – Kein Alias-Update.")

    # Logging in MLflow (als Run)
    try:
        with mlflow.start_run(run_name="validate_predictions") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "validate_models")
            mlflow.log_param("n_test_users", len(valid_users))
            mlflow.log_metric("precision_10_hybrid", hybrid_mean)
            mlflow.log_metric("precision_10_user", user_mean)
            score_df = pd.DataFrame({
                "user_id": valid_users,
                "hybrid_score": hybrid_scores,
                "user_score": user_scores
            })
            score_path = "/opt/airflow/data/processed/validation_scores.csv"
            score_df.to_csv(score_path, index=False)
            mlflow.log_artifact(score_path, artifact_path="validation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Logging in MLflow: {e}", exc_info=True)
        return

    logging.info("🎉 Validation & Registry update abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_user_count", type=int, default=100)
    args = parser.parse_args()
    validate_models(test_user_count=args.test_user_count)