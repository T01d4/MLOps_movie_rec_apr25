# src/movie/models/predict_best_model.py

import pandas as pd
import mlflow
from dotenv import load_dotenv
import os
import logging
import argparse
from mlflow.tracking import MlflowClient

load_dotenv()

REGISTRY_NAME = "hybrid_deep_model"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_artifact_df_from_best_model(model_name, artifact_rel_path):
    """
    Lädt ein CSV-Artifact (als DataFrame) direkt aus der besten Model-Version im MLflow Registry.
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "best_model")
    run_id = mv.run_id
    file_path = client.download_artifacts(run_id, artifact_rel_path)
    return pd.read_csv(file_path, index_col=0)

def load_artifact_pkl_from_best_model(model_name, artifact_rel_path):
    """
    Lädt ein Pickle-Artifact direkt aus der besten Model-Version im MLflow Registry.
    """
    import pickle
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "best_model")
    run_id = mv.run_id
    file_path = client.download_artifacts(run_id, artifact_rel_path)
    with open(file_path, "rb") as f:
        return pickle.load(f)

def predict_best_model(n_users=10):
    logging.info("🚀 Starte Prediction für hybrid_deep_model über MLflow Registry")
    # Modell (pyfunc, Wrapper) direkt aus Registry laden (wie immer)
    model_uri = f"models:/{REGISTRY_NAME}@best_model"
    model = mlflow.pyfunc.load_model(model_uri)

    # Embedding-CSV direkt als DataFrame aus Registry holen (kein Kopieren!)
    input_matrix = load_artifact_df_from_best_model(
        REGISTRY_NAME, "features/hybrid_deep_embedding.csv"
    )
    feature_count = input_matrix.shape[1]
    input_matrix.columns = [f"emb_{i}" for i in range(feature_count)]
    input_df = input_matrix.iloc[:n_users].copy().astype("float32")
    logging.info(f"📥 Embedding geladen (direkt aus Registry): Shape: {input_matrix.shape}")

    # Beispiel: Falls du noch einen echten Sklearn-KNN brauchst:
    # knn_model = load_artifact_pkl_from_best_model(REGISTRY_NAME, "backup_model/hybrid_deep_knn.pkl")

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
        logging.error(f"❌ Fehler bei der Modellvorhersage: {e}")
        raise

    # Ergebnis NUR im RAM!
    logging.info("✅ Prediction erfolgreich – keine Speicherung auf Disk, nur im RAM.")

    # Wenn du willst, kannst du result_df weiterreichen, zurückgeben oder sonst was
    print(result_df.head())  # Nur für Testzwecke

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=10)
    args = parser.parse_args()
    predict_best_model(n_users=args.n_users)