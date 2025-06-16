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
import getpass
import json


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validate")

DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models")

RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_deep_knn.pkl")
EMBEDDING_PATH = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
RATINGS_PATH = os.path.join(RAW_DIR, "ratings.csv")
BEST_EMBEDDING_PATH = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding_best.csv")
VALIDATION_SCORES_PATH = os.path.join(PROCESSED_DIR, "validation_scores_hybrid_deep.csv")
DVC_FILE = f"{BEST_EMBEDDING_PATH}.dvc"


def update_best_model_in_mlflow(precision, client, model_name, model_version):
    try:
        alias_version = client.get_model_version_by_alias(model_name, "best_model")
        best_version = alias_version.version
        best_run_id = alias_version.run_id
        old_run = client.get_run(best_run_id)
        best_prec = float(old_run.data.metrics.get("precision_10", 0))
        logging.info(f"Current best precision_10: {best_prec} (Version: {best_version})")
    except Exception as e:
        logging.warning(f"No best_model alias found: {e} -> Initializing best value to 0.0")
        best_prec = 0.0
        best_version = None

    if precision > best_prec:
        logging.info(f"🏆 New best score! {precision:.4f} > {best_prec:.4f} (Version: {model_version})")
        client.set_registered_model_alias(model_name, "best_model", model_version)
        logging.info(f"Alias 'best_model' set to version {model_version}!")

        try:
            if not os.path.exists(EMBEDDING_PATH):
                logging.error(f"❌ EMBEDDING_PATH does not exist: {EMBEDDING_PATH}")
                return
            if os.path.exists(BEST_EMBEDDING_PATH):
                os.remove(BEST_EMBEDDING_PATH)
            shutil.copy(EMBEDDING_PATH, BEST_EMBEDDING_PATH)
            logging.info("✅ Best-Embedding als _best gespeichert!")

            model_version_obj = client.get_model_version(model_name, model_version)
            train_run_id = model_version_obj.run_id

            mlflow.tracking.MlflowClient().log_artifact(
                run_id=train_run_id,
                local_path=BEST_EMBEDDING_PATH,
                artifact_path="best_embedding"
            )
            logging.info("✅ Best embedding logged as artifact in training run!")

            # === create and upload pipeline_conf_best.json ===
            original_conf = os.path.join(PROCESSED_DIR, "pipeline_conf.json")
            best_conf = os.path.join(PROCESSED_DIR, "pipeline_conf_best.json")

            if os.path.exists(original_conf):
                shutil.copy(original_conf, best_conf)
                logging.info("✅ pipeline_conf_best.json lokal gespeichert!")

                mlflow.tracking.MlflowClient().log_artifact(
                    run_id=train_run_id,
                    local_path=best_conf,
                    artifact_path="best_config"
                )
                logging.info("✅ pipeline_conf_best.json logged as artifact in training run!")
            else:
                logging.warning(f"⚠️ pipeline_conf.json not found at {original_conf}")

        except Exception as ex:
            logging.error(f"❌ Error while uploading best artifacts to MLflow: {ex}")

    else:
        logging.info(f"No new best score – precision not improved ({precision:.4f} <= {best_prec:.4f})")


    # Always update precision_10 tag for current model version
    client.set_model_version_tag(model_name, model_version, "precision_10", str(precision))

def get_latest_model_version(client, model_name):
    """Hole die Modellversion mit dem höchsten creation timestamp (=neuester Run)."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logging.error("❌ No model versions found!")
        return None, None
    # Sort by creation timestamp descending, return the first
    versions_sorted = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
    latest_version = versions_sorted[0]
    return latest_version.version, latest_version.run_id

def validate_deep_hybrid(test_user_count=100):
    logging.info("🚀 Starting validation (Deep Hybrid Only)")
    validation_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = "movie_recommendation_validation"
    val_task = "full_eval"

    try:
        ratings = pd.read_csv(RATINGS_PATH)
        embedding_df = pd.read_csv(EMBEDDING_PATH, index_col=0)
        knn_model = pickle.load(open(MODEL_PATH, "rb"))
        logging.info("📥 Deep hybrid model & embeddings loaded – starting evaluation")
    except Exception as e:
        logging.error(f"❌ Error loading data/models: {e}", exc_info=True)
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
            logging.warning(f"⚠️ Error with user {uid}: {e}")
            continue

    if not valid_users:
        logging.error("❌ No valid users for evaluation!")
        return

    precision_10 = float(np.mean(hybrid_scores))
    logging.info(f"📊 precision_10_hybrid_deep: {precision_10:.2f}")

    # --- MLflow Logging & Registry Best Model Update ---
    try:
        with mlflow.start_run(run_name=f"{experiment_name}_deep_hybrid") as run:
            mlflow.set_tag("experiment_name", experiment_name)
            mlflow.set_tag("validation_date", validation_date)
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "validate_models")
            mlflow.set_tag("val_task", val_task)
            mlflow.set_tag("model_type", "hybrid_deep_knn")
            mlflow.set_tag("test_user_count", test_user_count)
            mlflow.log_param("n_test_users", len(valid_users))
            mlflow.log_metric("precision_10", precision_10)

            score_df = pd.DataFrame({
                "user_id": valid_users,
                "hybrid_score": hybrid_scores,
            })
            score_path = VALIDATION_SCORES_PATH
            score_df.to_csv(score_path, index=False)
            mlflow.log_artifact(score_path, artifact_path="validation")

            # Get the latest model version
            client = MlflowClient()
            model_name = "hybrid_deep_model"
            current_version, _ = get_latest_model_version(client, model_name)
            if current_version:
                # Get run ID of the model version
                model_version_obj = client.get_model_version(model_name, current_version)
                train_run_id = model_version_obj.run_id

                # Set precision_10 metric on training run
                client.log_metric(run_id=train_run_id, key="precision_10", value=precision_10)

                # Update alias if it's a new best
                update_best_model_in_mlflow(precision_10, client, model_name, current_version)
            else:
                logging.warning("Could not determine current model version for comparison.")

    except Exception as e:
        logging.error(f"❌ Error during MLflow logging/alias update: {e}", exc_info=True)
        return

    logging.info("🎉 Validation complete.")
    # Nach logging.info("🎉 Validation complete.")
    try:
        PROM_FILE_PATH = os.getenv("REPORT_DIR", "/app/reports")
        os.makedirs(PROM_FILE_PATH, exist_ok=True)
        precision_file = os.path.join(PROM_FILE_PATH, "precision_metrics.prom")
        with open(precision_file, "w") as f:
            f.write(f'model_precision_at_10{{model="Deep Hybrid-KNN_best"}} {precision_10:.4f}\n')
        logging.info(f"💾 Prometheus precision_10 metric written to: {precision_file}")
    except Exception as e:
        logging.warning(f"⚠️ Could not write precision_10 prom file: {e}")
   # === Evidently Drift Detection + Prometheus Drift-Metriken ===
    try:

        current_df = pd.read_csv(EMBEDDING_PATH)
        reference_df = pd.read_csv(BEST_EMBEDDING_PATH) if os.path.exists(BEST_EMBEDDING_PATH) else current_df.copy()

        column_mapping = ColumnMapping()
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
        drift_json = report.as_dict()

        drift_alert = int(drift_json["metrics"][0]["result"]["dataset_drift"])
        drift_share = drift_json["metrics"][0]["result"]["share_of_drifted_columns"]

        # Prometheus-Metriken für Drift schreiben
        drift_file = os.path.join(PROM_FILE_PATH, "training_metrics.prom")
        with open(drift_file, "w") as f:
            f.write(f'model_drift_alert{{model="Deep Hybrid-KNN_best"}} {drift_alert}\n')
            f.write(f'data_drift_share{{model="Deep Hybrid-KNN_best"}} {drift_share:.4f}\n')
        logging.info("📈 Drift-Metriken für Prometheus geschrieben.")

    except Exception as e:
        logging.warning(f"⚠️ Evidently drift analysis failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_user_count", type=int, default=100)
    args = parser.parse_args()
    validate_deep_hybrid(
        test_user_count=args.test_user_count
    )