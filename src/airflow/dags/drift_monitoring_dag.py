# === src/airflow/dags/drift_monitoring_dag.py ===

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess, os, json
import logging

logger = logging.getLogger("airflow.task")

# === Konfigurationspfade ===
DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
REPORT_DIR = os.environ.get("REPORT_DIR", "/opt/airflow/reports")
MONITORING_CONF_PATH = os.path.join(DATA_DIR, "monitoring", "monitoring_conf.json")

# === Lade Config falls benötigt ===
if os.path.exists(MONITORING_CONF_PATH):
    with open(MONITORING_CONF_PATH, "r") as f:
        conf = json.load(f)
else:
    conf = {}

# === Default Args ===
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

# === Python Callables ===
def analyze_request_drift():
    logger.info("📥 Starte Request-Drift-Analyse...")
    result = subprocess.run(
        ["python", "/opt/airflow/src/monitoring/analyze_drift_requests.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    logger.info("✅ Drift-Skript ausgeführt mit exit code %s", result.returncode)
    if result.stdout:
        logger.info("📄 stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("⚠️ stderr:\n%s", result.stderr)


def generate_embedding_snapshot():
    logger.info("🧬 Generiere neues Snapshot-Embedding...")
    result = subprocess.run(
        ["python", "/opt/airflow/src/monitoring/generate_embedding.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    logger.info("✅ Embedding-Snapshot fertig mit Code %s", result.returncode)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)


def generate_extended_report():
    logger.info("📊 Erzeuge erweiterten Drift-Report...")
    result = subprocess.run(
        ["python", "/opt/airflow/src/monitoring/generate_drift_report_extended.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    logger.info("✅ Extended Report fertig mit Code %s", result.returncode)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)


# === DAG Definition ===
with DAG(
    dag_id="drift_monitoring_dag",
    default_args=default_args,
    start_date=datetime(2025, 6, 1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["monitoring"]
) as dag:

    analyze_snapshot_drift = BashOperator(
        task_id="analyze_snapshot_drift",
        bash_command="python /opt/airflow/src/monitoring/analyze_drift.py",
        env={"DATA_DIR": DATA_DIR, "REPORT_DIR": "/opt/airflow/reports"},
        do_xcom_push=False
    )

    analyze_request = PythonOperator(
        task_id="analyze_request_drift",
        python_callable=analyze_request_drift
    )

    snapshot_generation = PythonOperator(
        task_id="generate_embedding_snapshot",
        python_callable=generate_embedding_snapshot
    )

    generate_extended = PythonOperator(
        task_id="generate_drift_report_extended",
        python_callable=generate_extended_report
    )

    snapshot_generation >> analyze_snapshot_drift >> analyze_request >> generate_extended 