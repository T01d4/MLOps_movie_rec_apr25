# airflow/dags/train_deep_model_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import subprocess
import os
# import load_dotenv
# # === ENV laden ===
# load_dotenv.load_dotenv(load_dotenv.find_dotenv())

def run_and_log(command: list, cwd: str = "/opt/airflow"):
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

def run_import_raw_data():
    run_and_log(["python", "/opt/airflow/src/data/import_raw_data.py"])

def run_make_dataset():
    run_and_log(["python", "/opt/airflow/src/data/make_dataset.py"])

def run_build_features():
    run_and_log(["python", "/opt/airflow/src/features/build_features.py"])

def run_train_model():
    run_and_log(["python", "/opt/airflow/src/models/train_model.py"])

def run_train_deep_hybrid_model(**context):
    print(context)
    print(os.environ.get("MLFLOW_TRACKING_URI"))
    print(os.environ.get("DAGSHUB_USER"))
    print(str(os.environ.get("DAGSHUB_TOKEN"))[:10])
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    n_neighbors = conf.get("n_neighbors", 10)
    latent_dim = conf.get("latent_dim", 64)
    epochs = conf.get("epochs", 30)
    tfidf_features = conf.get("tfidf_features", 300)

    run_and_log([
        "python", "/opt/airflow/src/models/train_hybrid_deep_model.py",
        f"--n_neighbors={n_neighbors}",
        f"--latent_dim={latent_dim}",
        f"--epochs={epochs}",
        f"--tfidf_features={tfidf_features}"
    ])

def run_validate_model(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    test_user_count = conf.get("test_user_count", 100)
    run_and_log([
        "python", "/opt/airflow/src/models/validate_model.py",
        f"--test_user_count={test_user_count}"
    ])

def run_predict_best_model():
    run_and_log(["python", "/opt/airflow/src/models/predict_best_model.py"])

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

    import_raw_data >> make_dataset >> build_features
    build_features >> [train_model, train_deep_hybrid_model] >> validate >> predict
