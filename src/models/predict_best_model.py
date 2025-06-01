# src/models/predict_best_model.py

import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
import dagshub

# === Logging Setup für Airflow ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === ENV Setup ===
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

#dagshub.init(repo_owner='sacer11', repo_name='MLOps_movie_rec_apr25', mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

def predict_best_model():
    logging.info("🚀 Starte Vorhersage mit MLflow Registry-Modell")

    try:
        # === 1. Lade User-Matrix ===
        user_matrix_path = "/opt/airflow/data/processed/user_matrix.csv"
        user_matrix = pd.read_csv(user_matrix_path, index_col=0)
        logging.info(f"📥 User-Matrix geladen: {user_matrix_path} – Shape: {user_matrix.shape}")
    except Exception as e:
        logging.error(f"❌ Fehler beim Laden der User-Matrix: {e}")
        return

    try:
        # === 2. Lade Modell aus der Registry ===
        model_uri = "models:/movie_model@best_model"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"📦 Modell geladen aus MLflow Registry: {model_uri}")
        except mlflow.exceptions.MlflowException as e:
            logging.error(f"❌ Kein Modell mit Alias @best_model gefunden! Prüfe, ob das Modell korrekt registriert und der Alias gesetzt ist. Error: {e}")
            return
        except Exception as e:
            logging.error(f"❌ Fehler beim Laden des Modells aus MLflow: {e}")
            return

    except Exception as e:
        logging.error(f"❌ Fehler beim Initialisieren des Modells: {e}")
        return

    try:
        # === 3. Vorhersagen durchführen ===
        input_df = user_matrix.iloc[:10].copy()
        predictions = model.predict(input_df)

        if not isinstance(predictions, list):
            logging.error("❌ Modell hat kein gültiges Ausgabeformat zurückgegeben (Liste erwartet)")
            return

        result_df = pd.DataFrame({
            "user_id": input_df.index,
            "recommendations": predictions
        })

        out_path = "/opt/airflow/data/predictions.csv"
        result_df.to_csv(out_path, index=False)
        logging.info(f"💾 Vorhersagen gespeichert unter: {out_path}")

        with mlflow.start_run(run_name="predict_best_model") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "predict_best_model")
            mlflow.set_tag("model_uri", model_uri)
            mlflow.log_param("n_predicted_users", len(result_df))
            mlflow.log_artifact(out_path, artifact_path="recommendations")
        logging.info("✅ Predictions erfolgreich in MLflow geloggt")

    except Exception as e:
        logging.error(f"❌ Fehler bei der Vorhersage oder dem Logging: {e}")

if __name__ == "__main__":
    predict_best_model()