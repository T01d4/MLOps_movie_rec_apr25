# src/models/predict_best_model.py

import pandas as pd
import mlflow
from dotenv import load_dotenv
import os
import logging
import argparse
from mlflow.tracking import MlflowClient
import json

load_dotenv()

REGISTRY_NAME = "hybrid_deep_model"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_artifact_df_from_best_model(model_name, artifact_rel_path):
    """
    Loads a CSV artifact (as DataFrame) directly from the best model version in the MLflow registry.
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "best_model")
    run_id = mv.run_id
    file_path = client.download_artifacts(run_id, artifact_rel_path)
    return pd.read_csv(file_path, index_col=0)

def load_artifact_pkl_from_best_model(model_name, artifact_rel_path):
    """
    Loads a Pickle artifact directly from the best model version in the MLflow registry.
    """
    import pickle
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "best_model")
    run_id = mv.run_id
    file_path = client.download_artifacts(run_id, artifact_rel_path)
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_config_from_best_model(model_name):
    """
    Loads the pipeline_conf_best.json from the MLflow run with alias 'best_model'.
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "best_model")
    run_id = mv.run_id
    config_path = client.download_artifacts(run_id, "best_config/pipeline_conf_best.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_best_model(n_users=10):
    logging.info("🚀 Starte Prediction für hybrid_deep_model über MLflow Registry")
    # Load model (pyfunc, wrapper) directly from registry
    model_uri = f"models:/{REGISTRY_NAME}@best_model"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load embedding CSV as DataFrame directly from registry (no local copy!)
    input_matrix = load_artifact_df_from_best_model(
        REGISTRY_NAME, "best_embedding/hybrid_deep_embedding_best.csv"
    )
    feature_count = input_matrix.shape[1]
    input_matrix.columns = [f"emb_{i}" for i in range(feature_count)]
    input_df = input_matrix.iloc[:n_users].copy().astype("float32")
    logging.info(f"📥 Embedding loaded (directly from registry): Shape: {input_matrix.shape}")

    # Optional: Load sklearn KNN model if needed
    # knn_model = load_artifact_pkl_from_best_model(REGISTRY_NAME, "knn_model/knn_model.pkl")

    # Load config
    try:
        config = load_config_from_best_model(REGISTRY_NAME)
        logging.info(f"📄 Konfiguration geladen aus Registry:")
        for key, value in config.items():
            logging.info(f" - {key}: {value}")
    except Exception as e:
        logging.warning(f"⚠️ Could not load pipeline_conf.json: {e}")
        config = {}
    # Prediction
    try:
        predictions = model.predict(input_df)
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        result_df = pd.DataFrame({
            "user_id": input_df.index,
            "recommendations": predictions
        })
    except Exception as e:
        logging.error(f"❌ Error during model prediction: {e}")
        raise

    # Result is kept in memory only!
    logging.info("✅ Prediction successful – result is kept in RAM, not saved to disk.")

    # You can return, pass or further process result_df if needed
    print(result_df.head())  # For test purposes only

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=10)
    args = parser.parse_args()
    predict_best_model(n_users=args.n_users)