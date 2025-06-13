# === src/airflow/dags/drift_monitoring_dag.py ===

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import json, os

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
MONITORING_CONF_PATH = os.path.join(DATA_DIR, "monitoring", "monitoring_conf.json")

with open(MONITORING_CONF_PATH, "r") as f:
    conf = json.load(f)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    dag_id="drift_monitoring_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["monitoring"]
) as dag:

    analyze_drift = BashOperator(
        task_id="analyze_drift",
        bash_command="python /opt/airflow/src/monitoring/analyze_drift.py"
    )

    export_metrics = BashOperator(
        task_id="export_drift_metrics",
        bash_command="python /opt/airflow/src/monitoring/export_drift_metrics.py"
    )

    analyze_drift >> export_metrics