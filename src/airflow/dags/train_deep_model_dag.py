# airflow/dags/train_deep_model_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
import subprocess
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

def run_and_log(command: list, cwd: str = "/opt/airflow"):
    import subprocess
    try:
        logging.info(f"🟦 Running command: {' '.join(map(str, command))}")
        logging.info(f"🟦 Working directory: {cwd}")
        logging.info(f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        logging.info(f"User: {os.getenv('MLFLOW_TRACKING_USERNAME')}")
        logging.info(f"Password: {str(os.getenv('MLFLOW_TRACKING_PASSWORD'))[:10]}")
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
                logging.info("✅ No commit necessary (nothing to commit).")
                return
        if result.returncode != 0:
            logging.error(f"❌ Subprocess error (exit code {result.returncode}): {' '.join(map(str, command))}")
            raise subprocess.CalledProcessError(result.returncode, command)
    except Exception as e:
        logging.error(f"❌ Subprocess exception: {e}")
        raise

def run_import_raw_data():
    run_and_log(["python", "/opt/airflow/src/movie/data/import_raw_data.py"])

def run_make_dataset():
    run_and_log(["python", "/opt/airflow/src/movie/data/make_dataset.py"])  #

def run_build_features():
    run_and_log(["python", "/opt/airflow/src/movie/features/build_features.py"])

def run_train_model():
    run_and_log(["python", "/opt/airflow/src/movie/models/train_model.py"])

def run_train_deep_hybrid_model():
    run_and_log(["python", "/opt/airflow/src/movie/models/train_hybrid_deep_model.py"])

def run_validate_model(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    test_user_count = conf.get("test_user_count", 100)
    run_and_log([
        "python", "/opt/airflow/src/movie/models/validate_model.py",
        f"--test_user_count={test_user_count}"
    ])

def run_predict_best_model():
    run_and_log(["python", "/opt/airflow/src/movie/models/predict_best_model.py"])


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}

with DAG(
    dag_id="deep_models_pipeline",
    default_args=default_args,
    description='Train, Validate & Predict Deep User & Hybrid Models parallel',
    schedule_interval=None,
    catchup=False,
) as dag:
    import_raw_data = PythonOperator(
        task_id="import_raw_data",
        python_callable=run_import_raw_data
    )
    make_dataset = PythonOperator(
        task_id="make_dataset",
        python_callable=run_make_dataset,
    )
    build_features = PythonOperator(
        task_id="build_features",
        python_callable=run_build_features,
    )
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_train_model
    )
    train_deep_hybrid_model = PythonOperator(
        task_id='train_deep_hybrid_model',
        python_callable=run_train_deep_hybrid_model,
        provide_context=True
    )
    validate = PythonOperator(
        task_id="validate_models",
        python_callable=run_validate_model
    )
    predict = PythonOperator(
        task_id="predict_best_model",
        python_callable=run_predict_best_model
    )
    trigger_monitoring = TriggerDagRunOperator(
        task_id="trigger_drift_monitoring_dag",
        trigger_dag_id="drift_monitoring_dag",  
        wait_for_completion=True  # True waiting for result
    )

    # DAG flow (analogous to a classic pipeline)
    import_raw_data >> make_dataset >> build_features
    build_features >> [train_model, train_deep_hybrid_model] >> validate >> predict >> trigger_monitoring 
