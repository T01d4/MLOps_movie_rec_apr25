from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import os
import logging

def run_and_log(command: list, cwd:str="/opt/airflow"):
    import subprocess
    try:
        logging.info(f"🟦 Running command: {' '.join(map(str, command))}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        logging.info(f"🟩 [stdout]:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"🟨 [stderr]:\n{result.stderr}")
        if result.returncode == 1:
            output = (result.stdout or "") + (result.stderr or "")
            if ("nothing to commit" in output.lower() or "no changes added to commit" in output.lower()):
                logging.info("✅ Kein Commit nötig (nichts zu committen).")
                return
        if result.returncode != 0:
            logging.error(f"❌ Subprozess-Fehler (exit code {result.returncode}): {' '.join(map(str, command))}")
            raise subprocess.CalledProcessError(result.returncode, command)
    except Exception as e:
        logging.error(f"❌ Subprozess-Ausnahme: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

with DAG(
    dag_id="deep_models_pipeline",
    default_args=default_args,
    description='Train, Validate & Predict Deep User & Hybrid Models parallel',
    schedule_interval=None,
    catchup=False,
) as dag:
    
    train_deep_user_model = BashOperator(
        task_id='train_deep_user_model',
        bash_command='python /opt/airflow/src/models/train_user_deep_model.py'
    )

    train_deep_hybrid_model = BashOperator(
        task_id='train_deep_hybrid_model',
        bash_command='python /opt/airflow/src/models/train_hybrid_deep_model.py'
    )

    validate_deep_models = BashOperator(
        task_id='validate_deep_models',
        bash_command='python /opt/airflow/src/models/validate_deep_models.py'
    )

    predict_deep_models = BashOperator(
        task_id='predict_deep_models',
        bash_command='python /opt/airflow/src/models/predict_deep_models.py'
    )

    # Reihenfolge: Beide Trainings parallel → dann Validation → dann Prediction
    [train_deep_user_model, train_deep_hybrid_model] >> validate_deep_models >> predict_deep_models