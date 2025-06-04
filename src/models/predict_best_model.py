# src/models/predict_best_model.py

import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
import os
import logging
import pickle
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

MODEL_DIR = "/opt/airflow/models"

MODEL_FILE_PATHS = {
    "hybrid": os.path.join(MODEL_DIR, "hybrid_model.pkl"),
    "user": os.path.join(MODEL_DIR, "user_model.pkl"),
    "hybrid_dl": os.path.join(MODEL_DIR, "hybrid_deep_knn.pkl"),
    "user_dl": os.path.join(MODEL_DIR, "user_deep_knn.pkl")
}
MATRIX_PATHS = {
    "hybrid": ("/opt/airflow/data/processed/hybrid_matrix.csv", "/opt/airflow/data/processed/hybrid_matrix_features.txt"),
    "user": ("/opt/airflow/data/processed/user_matrix.csv", "/opt/airflow/data/processed/user_matrix_features.txt"),
    "hybrid_dl": ("/opt/airflow/data/processed/hybrid_deep_embedding.csv", None),
    "user_dl": ("/opt/airflow/data/processed/user_deep_embedding.csv", None)
}

def load_matrix(matrix_path, features_path=None):
    if features_path:
        with open(features_path, "r") as f:
            features = [line.strip() for line in f.readlines()]
        features_no_index = features[1:] if features[0].lower() in ["movieid", "userid"] else features
        df = pd.read_csv(matrix_path, index_col=0)
        assert list(df.columns) == features_no_index, (
            f"Spalten stimmen nicht! Matrix: {len(df.columns)} Features: {len(features_no_index)}"
        )
        return df
    else:
        return pd.read_csv(matrix_path, index_col=0)

def predict_best_model(model_type="hybrid", pipeline_type="classic", n_users=10):
    logging.info(f"🚀 Starte Prediction – Typ: {model_type}, Pipeline: {pipeline_type}")
    model_key = model_type if pipeline_type == "classic" else f"{model_type}_dl"
    try:
        model_path = MODEL_FILE_PATHS[model_key]
        matrix_path, features_path = MATRIX_PATHS[model_key]

        if not os.path.exists(model_path):
            logging.error(f"❌ Modell nicht lokal vorhanden: {model_path}")
            raise FileNotFoundError(f"Modell nicht vorhanden: {model_path}")
        model = pickle.load(open(model_path, "rb"))

        input_matrix = load_matrix(matrix_path, features_path)
        logging.info(f"📥 Eingabematrix geladen: {matrix_path} – Shape: {input_matrix.shape}")

        # Limitiere ggf. Nutzer/Filme
        input_df = input_matrix.iloc[:n_users].copy()

        if hasattr(model, "kneighbors"):
            _, predictions = model.kneighbors(input_df)
        else:
            logging.error("❌ Geladenes Modell hat keine kneighbors-Methode!")
            raise AttributeError("Geladenes Modell hat keine kneighbors-Methode!")

        result_df = pd.DataFrame({
            "user_id": input_df.index,
            "recommendations": predictions.tolist()
        })

        out_path = f"/opt/airflow/data/processed/predictions_{model_key}.csv"
        result_df.to_csv(out_path, index=False)
        logging.info(f"💾 Vorhersagen gespeichert unter: {out_path}")

        with mlflow.start_run(run_name=f"predict_best_model_{model_key}") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "predict_best_model")
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("pipeline_type", pipeline_type)
            mlflow.set_tag("model_path", model_path)
            mlflow.log_param("n_predicted_users", len(result_df))
            mlflow.log_artifact(out_path, artifact_path="recommendations")
        logging.info("✅ Predictions erfolgreich in MLflow geloggt")

    except Exception as e:
        logging.error(f"❌ Fehler bei der Vorhersage oder dem Logging: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["hybrid", "user"])
    parser.add_argument("--pipeline_type", type=str, default="classic", choices=["classic", "dl"])
    parser.add_argument("--n_users", type=int, default=10)
    args = parser.parse_args()
    predict_best_model(model_type=args.model_type, pipeline_type=args.pipeline_type, n_users=args.n_users)