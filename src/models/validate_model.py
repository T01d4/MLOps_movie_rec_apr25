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
from datetime import datetime
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validate")

MODEL_DIR = "/opt/airflow/models"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_deep_knn.pkl")
EMBEDDING_PATH = "/opt/airflow/data/processed/hybrid_deep_embedding.csv"

def update_best_model_in_mlflow(precision, client, model_name, model_version):
    # 1. Aktuellen best_model-Wert aus der Registry holen (MLflow)
    try:
        alias_version = client.get_model_version_by_alias(model_name, "best_model")
        best_version = alias_version.version
        best_run_id = alias_version.run_id
        old_run = client.get_run(best_run_id)
        best_prec = float(old_run.data.metrics.get("precision_10", 0))
        logging.info(f"Aktueller Bestwert precision_10: {best_prec} (Version: {best_version})")
    except Exception as e:
        logging.warning(f"Kein best_model-Alias gefunden: {e} -> Initialisiere Bestwert mit 0.0")
        best_prec = 0.0
        best_version = None

    # 2. Vergleich und ggf. neuen Bestwert setzen
    if precision > best_prec:
        logging.info(f"🏆 Neuer Bestwert! {precision:.4f} > {best_prec:.4f} (Version: {model_version})")
        client.set_registered_model_alias(model_name, "best_model", model_version)
        logging.info(f"Alias 'best_model' wurde auf Version {model_version} gesetzt!")

        # ==== NUR das Featurefile/Embedding speichern und mit DVC versionieren ====
        try:
            best_embedding_path = "/opt/airflow/data/processed/hybrid_deep_embedding_best.csv"
            shutil.copy(EMBEDDING_PATH, best_embedding_path)
            logging.info("✅ Best-Embedding als _best gespeichert!")

            # DVC add
            subprocess.run(["dvc", "add", best_embedding_path], check=True)
            # Git add/commit für DVC-File und .gitignore!
            subprocess.run(["git", "add", f"{best_embedding_path}.dvc", ".gitignore"], check=True)
            subprocess.run([
                "git", "commit", "-m", f"Track new best embedding (precision={precision:.4f})"
            ], check=True)
            # DVC push
            subprocess.run(["dvc", "push"], check=True)
            logging.info("✅ DVC add, git commit & push für Best-Embedding abgeschlossen!")
        except Exception as e:
            logging.error(f"❌ Fehler beim Kopieren oder DVC push der _best-Embedding-Datei: {e}")
    else:
        logging.info(f"Kein Bestwert – Präzision nicht verbessert ({precision:.4f} <= {best_prec:.4f})")

    # 3. **Immer** Tag für precision_10 auf aktuelle Modellversion updaten!
    client.set_model_version_tag(model_name, model_version, "precision_10", str(precision))

def get_latest_model_version(client, model_name):
    """Hole die Modellversion mit dem höchsten creation timestamp (=neuester Run)."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logging.error("❌ Keine Modellversionen gefunden!")
        return None, None
    # Sortiere nach creation_timestamp absteigend, nimm den ersten
    versions_sorted = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
    latest_version = versions_sorted[0]
    return latest_version.version, latest_version.run_id

def validate_deep_hybrid(test_user_count=100):
    logging.info("🚀 Starte Validierung (Deep Hybrid Only)")
    validation_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = "movie_recommendation_validation"
    val_task = "full_eval"

    try:
        ratings = pd.read_csv("/opt/airflow/data/raw/ratings.csv")
        embedding_df = pd.read_csv(EMBEDDING_PATH, index_col=0)
        knn_model = pickle.load(open(MODEL_PATH, "rb"))
        logging.info("📥 Deep Hybrid Model & Embeddings geladen – Beginne Evaluation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Laden der Daten/Modelle: {e}", exc_info=True)
        return

    test_users = embedding_df.index[:test_user_count]
    hybrid_scores, valid_users = [], []

    for uid in test_users:
        try:
            uvec = embedding_df.loc[uid].values.reshape(1, -1)
            if uvec.shape[1] != knn_model.n_features_in_:
                raise ValueError(f"Modell erwartet {knn_model.n_features_in_} Features, hat aber {uvec.shape[1]}")
            _, idxs = knn_model.kneighbors(uvec)
            rec_movie_ids = embedding_df.index[idxs[0]]
            hit = ratings[(ratings["userId"] == int(uid)) & (ratings["movieId"].isin(rec_movie_ids))]
            hybrid_scores.append(1 if not hit.empty else 0)
            valid_users.append(uid)
        except Exception as e:
            logging.warning(f"⚠️ Fehler bei User {uid}: {e}")
            continue

    if not valid_users:
        logging.error("❌ Keine gültigen Nutzer zur Auswertung!")
        return

    precision_10 = float(np.mean(hybrid_scores))
    logging.info(f"📊 precision_10_hybrid_deep: {precision_10:.2f}")

    # --- MLflow Logging & Registry-Bestwert-Update ---
    try:
        with mlflow.start_run(run_name=f"{experiment_name}_deep_hybrid") as run:
            mlflow.set_tag("experiment_name", experiment_name)
            mlflow.set_tag("validation_date", validation_date)
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "validate_models")
            mlflow.set_tag("val_task", val_task)
            mlflow.set_tag("model_type", "hybrid_deep_knn")
            mlflow.log_param("n_test_users", len(valid_users))
            mlflow.log_metric("precision_10", precision_10)

            score_df = pd.DataFrame({
                "user_id": valid_users,
                "hybrid_score": hybrid_scores,
            })
            score_path = f"/opt/airflow/data/processed/validation_scores_hybrid_deep.csv"
            score_df.to_csv(score_path, index=False)
            mlflow.log_artifact(score_path, artifact_path="validation")

            # Hole die aktuellste Modellversion
            client = MlflowClient()
            model_name = "hybrid_deep_model"
            current_version, _ = get_latest_model_version(client, model_name)
            if current_version:
                # Hole Run-ID dieser Modellversion
                model_version_obj = client.get_model_version(model_name, current_version)
                train_run_id = model_version_obj.run_id

                # Setze die precision_10 als Metric im Trainings-Run
                client.log_metric(run_id=train_run_id, key="precision_10", value=precision_10)

                # Setze ggf. Alias wie gehabt
                update_best_model_in_mlflow(precision_10, client, model_name, current_version)
            else:
                logging.warning("Konnte aktuelle Modellversion für Bestwertvergleich nicht finden.")

    except Exception as e:
        logging.error(f"❌ Fehler beim Logging/Alias-Update in MLflow: {e}", exc_info=True)
        return

    logging.info("🎉 Validation abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_user_count", type=int, default=100)
    args = parser.parse_args()
    validate_deep_hybrid(
        test_user_count=args.test_user_count
    )