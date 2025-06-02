# src/models/predict_best_model.py

import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
from mlflow.tracking import MlflowClient
import pickle

# === Logging Setup für Airflow ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === ENV Setup ===
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

def get_best_model_type():
    client = MlflowClient()
    model_name = "movie_model"
    try:
        mv = client.get_model_version_by_alias(model_name, "best_model")
        model_type = mv.tags.get("type", "hybrid")
        if model_type is not None:
            model_type = model_type.strip()
        logging.info(f"🔎 Bestes Modell laut Registry: Typ={model_type!r}, Version={mv.version}")
        return model_type
    except Exception as e:
        logging.error(f"❌ Fehler beim Abrufen des best_model-Alias: {e}")
        return None

def load_matrix_with_features(matrix_path, features_path):
    with open(features_path, "r") as f:
        features = [line.strip() for line in f.readlines()]
    # MovieId als Index (wird _nicht_ als Feature übergeben!)
    return pd.read_csv(matrix_path, names=features, header=0, index_col=0)

def predict_best_model():
    logging.info("🚀 Starte Vorhersage mit MLflow Registry & DVC-Modell")
    try:
        # === 1. Bestes Modell bestimmen ===
        model_type = get_best_model_type()
        if model_type is None:
            logging.error("❌ Kein Modelltyp aus der Registry ermittelbar.")
            raise RuntimeError("Kein Modelltyp aus der Registry ermittelbar.")

        model_path = f"/opt/airflow/models/{model_type}_model.pkl"

        # === 2. Modell laden ===
        if not os.path.exists(model_path):
            logging.error(f"❌ Modell nicht lokal vorhanden: {model_path} – DVC Pull fehlt!")
            raise FileNotFoundError(f"Modell nicht lokal vorhanden: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"📦 Modell geladen: {model_path}")

        # === 3. Passende Feature-Matrix laden ===
        if model_type == "hybrid":
            input_matrix_path = "/opt/airflow/data/processed/hybrid_matrix.csv"
            features_path = "/opt/airflow/data/processed/hybrid_matrix_features.txt"
        elif model_type == "user":
            input_matrix_path = "/opt/airflow/data/processed/user_matrix.csv"
            features_path = "/opt/airflow/data/processed/user_matrix_features.txt"
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")

        input_matrix = load_matrix_with_features(input_matrix_path, features_path)
        logging.info(f"📥 Eingabematrix geladen: {input_matrix_path} – Shape: {input_matrix.shape}")

        # === 4. Vorhersagen durchführen ===
        input_df = input_matrix.iloc[:10].copy()
        if hasattr(model, "kneighbors"):
            _, predictions = model.kneighbors(input_df)
        else:
            logging.error("❌ Geladenes Modell hat keine kneighbors-Methode!")
            raise AttributeError("Geladenes Modell hat keine kneighbors-Methode!")

        result_df = pd.DataFrame({
            "user_id": input_df.index,
            "recommendations": predictions.tolist()
        })

        out_path = "/opt/airflow/data/predictions.csv"
        result_df.to_csv(out_path, index=False)
        logging.info(f"💾 Vorhersagen gespeichert unter: {out_path}")

        with mlflow.start_run(run_name="predict_best_model") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "predict_best_model")
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("model_path", model_path)
            mlflow.log_param("n_predicted_users", len(result_df))
            mlflow.log_artifact(out_path, artifact_path="recommendations")
        logging.info("✅ Predictions erfolgreich in MLflow geloggt")

    except Exception as e:
        logging.error(f"❌ Fehler bei der Vorhersage oder dem Logging: {e}")
        raise

if __name__ == "__main__":
    predict_best_model()