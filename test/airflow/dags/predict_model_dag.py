# airflow/dags/predict_model_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
import logging
from src.visualization.validate_predictions import validate_predictions
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(dotenv_path="/opt/airflow/.env")

def run_prediction():
    try:
        # MLflow Auth
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

        input_path = "/opt/airflow/data/input.csv"
        output_dir = "/opt/airflow/data/predictions"
        output_path = os.path.join(output_dir, "predicted_titles.csv")
        prediction_matrix = os.path.join(output_dir, "predictions.csv")

        model_path = "/opt/airflow/models/model.pkl"
        columns_path = "/opt/airflow/model_cache/columns.pkl"
        movie_ids_path = "/opt/airflow/models/model_ids.pkl"

        logging.info("📦 Lade Modell aus pickle-Datei...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logging.info("📑 Lade erwartete Spalten für Feature-Matching...")
        with open(columns_path, "rb") as f:
            expected_columns = pickle.load(f)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Eingabedatei fehlt: {input_path}")

        df = pd.read_csv(input_path)

        # 🧠 Stelle sicher, dass die Spalten stimmen
        df = df[expected_columns]

        # 🧮 Nearest Neighbors Prediction
        _, indices = model.kneighbors(df)

        # 🔁 Mapping der Indizes zu movieId
        if not os.path.exists(movie_ids_path):
            raise FileNotFoundError(f"Mapping-Datei fehlt: {movie_ids_path}")

        with open(movie_ids_path, "rb") as f:
            movie_ids = pickle.load(f)

        predictions = [[movie_ids[i] for i in row] for row in indices]

        # 💾 Speichern
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(predictions).to_csv(prediction_matrix, index=False, header=False)

        # ✅ Korrektes Format für validate_predictions: eine ID pro Zeile
        df_titles = pd.DataFrame(predictions[0], columns=["predicted_title"])
        df_titles.to_csv(output_path, index=False)

        logging.info(f"✅ Vorhersagen gespeichert unter: {output_path}")

        # ✅ Validierung starten
        validate_predictions(output_path)

    except Exception as e:
        logging.error(f"❌ Fehler bei der Vorhersage: {e}")
        raise

default_args = {
    "start_date": datetime(2023, 1, 1),
    "catchup": False
}

with DAG(
    dag_id="predict_model_dag",
    schedule_interval="@daily",
    default_args=default_args,
    description="Tägliche Vorhersage mit MLflow-Modell",
    tags=["mlflow", "prediction"]
) as dag:

    predict_task = PythonOperator(
        task_id="run_prediction",
        python_callable=run_prediction
    )