# src/models//predict_deep_models.py

import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

def predict_hybrid_deep_model():
    model_path = "/opt/airflow/models/hybrid_deep_knn.pkl"
    embedding_path = "/opt/airflow/data/processed/hybrid_deep_embedding.csv"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        embeddings = pd.read_csv(embedding_path, index_col=0)
        input_df = embeddings.iloc[:10]
        if hasattr(model, "kneighbors"):
            _, rec = model.kneighbors(input_df)
        else:
            raise AttributeError("KNN fehlt!")
        result_df = pd.DataFrame({"movie_id": input_df.index, "recommendations": rec.tolist()})
        out_path = "/opt/airflow/data/predictions_hybrid_deep.csv"
        result_df.to_csv(out_path, index=False)
        with mlflow.start_run(run_name="predict_hybrid_deep_model") as run:
            mlflow.set_tag("model_type", "hybrid_deep_knn")
            mlflow.log_param("n_predicted_movies", len(result_df))
            mlflow.log_artifact(out_path, artifact_path="recommendations")
        logging.info("✅ Deep Predictions in MLflow geloggt")
    except Exception as e:
        logging.error(f"❌ Fehler: {e}")

if __name__ == "__main__":
    predict_hybrid_deep_model()