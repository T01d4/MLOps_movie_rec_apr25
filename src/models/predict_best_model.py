# src/models/predict_best_model.py

import pandas as pd
import mlflow
from dotenv import load_dotenv
import os
import logging
import argparse
import subprocess

def ensure_best_files_exist():
    # Prüfe, ob das Best-Embedding existiert, sonst DVC pull
    best_embedding_path = "/opt/airflow/data/processed/hybrid_deep_embedding_best.csv"
    if not os.path.exists(best_embedding_path):
        logging.warning(f"Best-Embedding fehlt: {best_embedding_path} – Starte dvc pull ...")
        try:
            subprocess.run(["dvc", "pull", best_embedding_path], check=True)
            logging.info("DVC pull abgeschlossen.")
        except Exception as e:
            logging.error(f"❌ DVC pull fehlgeschlagen: {e}")
            raise
        # Check, ob Datei nach dvc pull vorhanden ist
        if not os.path.exists(best_embedding_path):
            raise FileNotFoundError(f"Nach DVC pull fehlt immer noch: {best_embedding_path}")
    return best_embedding_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validate")

REGISTRY_NAME = "hybrid_deep_model"
EMBEDDING_PATH = "/opt/airflow/data/processed/hybrid_deep_embedding_best.csv"

def predict_best_model(n_users=10):
    ensure_best_files_exist()
    logging.info("🚀 Starte Prediction für hybrid_deep_model über MLflow Registry")
    # --- Modell aus Registry laden ---
    model_uri = f"models:/{REGISTRY_NAME}@best_model"
    logging.info(f"📦 Lade Modell: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logging.error(f"❌ Konnte Modell nicht laden: {e}")
        raise

    # --- Eingabematrix laden ---
    input_matrix = pd.read_csv(EMBEDDING_PATH, index_col=0)
    feature_count = input_matrix.shape[1]
    input_matrix.columns = [f"emb_{i}" for i in range(feature_count)]
    input_df = input_matrix.iloc[:n_users].copy().astype("float32")
    logging.info(f"📥 Embedding geladen: {EMBEDDING_PATH} – Shape: {input_matrix.shape}")

    # --- Prediction für die ersten n_users ---
    input_df = input_matrix.iloc[:n_users].copy()
    # *** WICHTIG: Typ auf float32 casten ***
    input_df = input_df.astype("float32")

    try:
        predictions = model.predict(input_df)
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        result_df = pd.DataFrame({
            "user_id": input_df.index,
            "recommendations": predictions
        })
    except Exception as e:
        logging.error(f"❌ Fehler bei der Modellvorhersage: {e}")
        raise

    # --- Logging ---
    out_path = "/opt/airflow/data/processed/predictions_hybrid_deep_model.csv"
    result_df.to_csv(out_path, index=False)
    logging.info(f"💾 Vorhersagen gespeichert unter: {out_path}")

    with mlflow.start_run(run_name="predict_best_model_hybrid_deep") as run:
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("task", "predict_best_model")
        mlflow.set_tag("model_type", "hybrid_deep_model")
        mlflow.set_tag("model_registry", REGISTRY_NAME)
        mlflow.log_param("n_predicted", len(result_df))
        mlflow.log_artifact(out_path, artifact_path="recommendations")
    logging.info("✅ Predictions erfolgreich in MLflow geloggt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=10)
    args = parser.parse_args()
    predict_best_model(n_users=args.n_users)