from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)

# Funktion zum Erzeugen der Spalten-Pickle-Datei und der movieId-Mapping-Datei
def create_columns_and_ids():
    train_data_path = "/opt/airflow/data/processed/movies_matrix.csv"
    columns_output_path = "/opt/airflow/model_cache/columns.pkl"
    model_ids_output_path = "/opt/airflow/models/model_ids.pkl"

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Trainingsdaten nicht gefunden: {train_data_path}")

    df = pd.read_csv(train_data_path)
    columns = list(df.columns)

    # Speichere Spaltennamen
    os.makedirs(os.path.dirname(columns_output_path), exist_ok=True)
    with open(columns_output_path, "wb") as f:
        pickle.dump(columns, f)
    logging.info(f"✅ columns.pkl gespeichert unter: {columns_output_path}")
    logging.info(f"📊 Enthaltene Spalten: {columns}")

    # Speichere movieId-Mapping
    if "movieId" in df.columns:
        movie_ids = df["movieId"].tolist()
    else:
        movie_ids = df.index.tolist()  # Fallback: Index als ID

    os.makedirs(os.path.dirname(model_ids_output_path), exist_ok=True)
    with open(model_ids_output_path, "wb") as f:
        pickle.dump(movie_ids, f)
    logging.info(f"✅ model_ids.pkl gespeichert unter: {model_ids_output_path}")
    logging.info(f"🎬 Anzahl Filme im Mapping: {len(movie_ids)}")

# DAG Definition
default_args = {
    "start_date": datetime(2023, 1, 1),
    "catchup": False,
}

with DAG(
    dag_id="generate_columns_dag",
    schedule_interval=None,
    default_args=default_args,
    description="Erzeuge columns.pkl und model_ids.pkl",
    tags=["feature_engineering", "columns", "mapping"],
) as dag:

    generate_columns_task = PythonOperator(
        task_id="generate_columns",
        python_callable=create_columns_and_ids
    )
