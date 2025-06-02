# airflow/dags/mlops_movie_dag.py

from airflow import DAG
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

# --- Tasks mit Parameterzugriff ---



def run_train_hybrid_model(**context):
    conf = context['dag_run'].conf or {}
    n_neighbors = conf.get("n_neighbors", 10)
    tfidf_features = conf.get("tfidf_features", 300)
    run_and_log([
        "python", "/opt/airflow/src/models/train_hybrid_model.py",
        f"--n_neighbors={n_neighbors}",
        f"--tfidf_features={tfidf_features}"
    ])

def run_validate_models(**context):
    conf = context['dag_run'].conf or {}
    test_user_count = conf.get("test_user_count", 100)
    run_and_log([
        "python", "/opt/airflow/src/models/validate_model.py",
        f"--test_user_count={test_user_count}",
    ])

def git_has_staged_changes(repo_path="/opt/airflow"):
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())

def cleanup_all():
    artifacts = [
        "/opt/airflow/models/hybrid_model.pkl",
        "/opt/airflow/data/processed/hybrid_matrix.csv",
        "/opt/airflow/data/processed/hybrid_matrix_features.txt",
        "/opt/airflow/models/user_model.pkl",
        "/opt/airflow/data/processed/user_matrix.csv",
        "/opt/airflow/data/processed/user_matrix_features.txt"
    ]
    # Remove lock if exists
    dvc_lock = '/opt/airflow/.dvc/tmp/lock'
    if os.path.exists(dvc_lock):
        os.remove(dvc_lock)
    # Remove all artifacts and their .dvc files (nur wenn sie existieren)
    for art in artifacts:
        if os.path.exists(art):
            os.remove(art)
        dvc_file = art + ".dvc"
        if os.path.exists(dvc_file):
            os.remove(dvc_file)
            # Nur wenn die .dvc wirklich existiert, dann auch `dvc remove`
            try:
                run_and_log(["dvc", "remove", dvc_file])
            except Exception as e:
                logging.warning(f"Ignoriere Fehler bei dvc remove für {dvc_file}: {e}")

def dvc_add_and_push_all():
    # --- Lock-File-Schutz ---
    dvc_lock = '/opt/airflow/.dvc/tmp/lock'
    if os.path.exists(dvc_lock):
        os.remove(dvc_lock)
    # --- Git "safe.directory" und Identität setzen ---
    run_and_log(["git", "config", "--global", "--add", "safe.directory", "/opt/airflow"])
    run_and_log(["git", "config", "--global", "user.email", "airflow@mlops.local"])
    run_and_log(["git", "config", "--global", "user.name", "Airflow"])
    # --- Alle Artefakte für beide Modelle ---
    artifacts = [
        "/opt/airflow/models/hybrid_model.pkl",
        "/opt/airflow/data/processed/hybrid_matrix.csv",
        "/opt/airflow/data/processed/hybrid_matrix_features.txt",
        "/opt/airflow/models/user_model.pkl",
        "/opt/airflow/data/processed/user_matrix.csv",
        "/opt/airflow/data/processed/user_matrix_features.txt"
    ]
    for art in artifacts:
        run_and_log(["dvc", "add", art])
    run_and_log(["git", "add"] + [a + ".dvc" for a in artifacts])
    if git_has_staged_changes("/opt/airflow"):
        run_and_log(["git", "commit", "-m", "Automatisches DVC-Tracking (beide Modelle/Matrizen)"])
    else:
        logging.info("✅ Nichts zu committen (Skip git commit)")
    run_and_log(["dvc", "push"])

    
# ==== Standard-Tasks bleiben gleich ====
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

#def run_train_hybrid_model():
#    run_and_log(["python", "/opt/airflow/src/models/train_hybrid_model.py"])

#def run_validate_models():
#    run_and_log(["python", "/opt/airflow/src/models/validate_model.py"])

def run_predict_best_model():
    run_and_log(["python", "/opt/airflow/src/models/predict_best_model.py"])

def debug_env():
    logging.info(f"📡 URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    logging.info(f"👤 USER: {os.getenv('MLFLOW_TRACKING_USERNAME')}")
    logging.info(f"🔐 PW gesetzt: {bool(os.getenv('MLFLOW_TRACKING_PASSWORD'))}")

def dvc_pull_all():
    to_pull = [
        "/opt/airflow/models/hybrid_model.pkl",
        "/opt/airflow/data/processed/hybrid_matrix.csv",
        "/opt/airflow/data/processed/hybrid_matrix_features.txt",
        "/opt/airflow/models/user_model.pkl",
        "/opt/airflow/data/processed/user_matrix.csv",
        "/opt/airflow/data/processed/user_matrix_features.txt"
    ]
    for art in to_pull:
        run_and_log(["dvc", "pull", art])

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}

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
        python_callable=run_make_dataset,
    )
    build_features = PythonOperator(
        task_id="build_features",
        python_callable=run_build_features,
    )
    cleanup_all_task = PythonOperator(
        task_id="cleanup_all",
        python_callable=cleanup_all
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
        python_callable=run_train_hybrid_model,
        provide_context=True
    )
    dvc_add_push_all_task = PythonOperator(
        task_id="dvc_add_and_push_all",
        python_callable=dvc_add_and_push_all
    )
    debug_task = PythonOperator(
        task_id="debug_env",
        python_callable=debug_env
    )
    dvc_pull_task = PythonOperator(
        task_id="dvc_pull_all",
        python_callable=dvc_pull_all
    )
    validate = PythonOperator(
        task_id="validate_models",
        python_callable=run_validate_models,
        provide_context=True
    )
    predict = PythonOperator(
        task_id="predict_best_model",
        python_callable=run_predict_best_model
    )

    # === Task-Reihenfolge (schlank & sicher) ===
    import_raw_data >> make_dataset >> build_features
    build_features >> cleanup_all_task
    cleanup_all_task >> [train_model, train_user, train_hybrid]
    [train_user, train_hybrid] >> dvc_add_push_all_task
    dvc_add_push_all_task >> debug_task >> dvc_pull_task >> validate >> predict