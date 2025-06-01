# airflow/dags/mlops_movie_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import os
import logging
from airflow.operators.python import PythonOperator

# === Hilfsfunktion mit Log-Erfassung ===
def run_and_log(command: list):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logging.info(result.stdout)

# === Wrapper-Funktionen (nun mit Logging) ===
def run_import_raw_data():
    run_and_log(["python", "/opt/airflow/src/data/import_raw_data.py"])

def run_make_dataset():
    run_and_log(["python", "/opt/airflow/src/data/make_dataset.py"])

def run_build_features():
    run_and_log(["python", "/opt/airflow/src/features/build_features.py"])

def run_train_model():
    run_and_log(["python", "/opt/airflow/src/models/train_model.py"])

def run_train_user_model():
    run_and_log(["python", "/opt/airflow/src/models/train_user_model.py"])

def run_train_hybrid_model():
    run_and_log(["python", "/opt/airflow/src/models/train_hybrid_model.py"])

def run_validate_models():
    run_and_log(["python", "/opt/airflow/src/models/validate_model.py"])

def airflow_check_registry(**kwargs):
    # Einfach die Funktion importieren und ausführen!
    from src.models.checkreg import check_registry
    check_registry()


def run_predict_best_model():
    run_and_log(["python", "/opt/airflow/src/models/predict_best_model.py"])

def debug_env():
    logging.info(f"📡 URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    logging.info(f"👤 USER: {os.getenv('MLFLOW_TRACKING_USERNAME')}")
    logging.info(f"🔐 PW gesetzt: {bool(os.getenv('MLFLOW_TRACKING_PASSWORD'))}")

# === Default Args inkl. Retry-Konfiguration ===
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}

# === DAG Definition ===
with DAG(
    dag_id="movie_recommendation_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "movie"]
) as dag:

    import_raw_data = PythonOperator(
        task_id="import_raw_data",
        python_callable=run_import_raw_data
    )

    make_dataset = PythonOperator(
        task_id="make_dataset",
        python_callable=run_make_dataset
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=run_build_features
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_train_model
    )

    train_user = PythonOperator(
        task_id="train_user_model",
        python_callable=run_train_user_model
    )

    train_hybrid = PythonOperator(
        task_id="train_hybrid_model",
        python_callable=run_train_hybrid_model
    )

    debug_task = PythonOperator(
        task_id="debug_env",
        python_callable=debug_env
    )

    validate = PythonOperator(
        task_id="validate_models",
        python_callable=run_validate_models
    )

    check_registry_task = PythonOperator(
        task_id='check_model_registry',
        python_callable=airflow_check_registry,
        dag=dag,
    )


    predict = PythonOperator(
        task_id="predict_best_model",
        python_callable=run_predict_best_model
    )

    # === Task-Reihenfolge ===
    import_raw_data >> make_dataset >> build_features
    build_features >> [train_model, train_hybrid, train_user] >> debug_task >> validate >> check_registry_task >> predict