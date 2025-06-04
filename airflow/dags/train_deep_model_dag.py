# airflow/dags/train_deep_model_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import os
import logging

def run_and_log(command: list, cwd: str = "/opt/airflow"):
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

def run_train_deep_user_model(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    n_neighbors = conf.get("n_neighbors", 10)
    latent_dim = conf.get("latent_dim", 32)
    run_and_log([
        "python", "/opt/airflow/src/models/train_user_deep_model.py",
        f"--n_neighbors={n_neighbors}",
        f"--latent_dim={latent_dim}"
    ])

def run_train_deep_hybrid_model(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    n_neighbors = conf.get("n_neighbors", 10)
    latent_dim = conf.get("latent_dim", 32)
    run_and_log([
        "python", "/opt/airflow/src/models/train_hybrid_deep_model.py",
        f"--n_neighbors={n_neighbors}",
        f"--latent_dim={latent_dim}"
    ])

def run_validate_deep_models(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    test_user_count = conf.get("test_user_count", 100)
    run_and_log([
        "python", "/opt/airflow/src/models/validate_model.py",
        "--pipeline_type=dl",
        f"--test_user_count={test_user_count}"
    ])

def run_predict_best_model_hybrid_dl(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    n_users = conf.get("n_users", 100)
    run_and_log([
        "python", "/opt/airflow/src/models/predict_best_model.py",
        "--model_type=hybrid",
        "--pipeline_type=dl",
        f"--n_users={n_users}"
    ])

def run_predict_best_model_user_dl(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    n_users = conf.get("n_users", 100)
    run_and_log([
        "python", "/opt/airflow/src/models/predict_best_model.py",
        "--model_type=user",
        "--pipeline_type=dl",
        f"--n_users={n_users}"
    ])

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
    
    train_deep_user_model = PythonOperator(
        task_id='train_deep_user_model',
        python_callable=run_train_deep_user_model,
        provide_context=True
    )

    train_deep_hybrid_model = PythonOperator(
        task_id='train_deep_hybrid_model',
        python_callable=run_train_deep_hybrid_model,
        provide_context=True
    )

    validate = PythonOperator(
        task_id="validate_models",
        python_callable=run_validate_deep_models,
        provide_context=True
    )
    predict_hybrid = PythonOperator(
        task_id="predict_best_model_hybrid",
        python_callable=run_predict_best_model_hybrid_dl,
        provide_context=True
    )
    predict_user = PythonOperator(
        task_id="predict_best_model_user",
        python_callable=run_predict_best_model_user_dl,
        provide_context=True
    )

    [train_deep_user_model, train_deep_hybrid_model] >> validate >> [predict_hybrid, predict_user]