#airflow/dags/bento_api_pipeline.py
from airflow import DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests
import logging
import os
import json


def log_bento_response(response, logger, task_name=""):
    import os
    def log_multiline(text, level="info"):
        if not text:
            return
        for line in str(text).splitlines():
            if line.strip():
                getattr(logger, level)(f"BentoML [{task_name}] {line}")
    if isinstance(response, dict):
        log_multiline(response.get("stdout"), "info")
        log_multiline(response.get("stderr"), "warning")
        if response.get("msg"):
            log_multiline(response["msg"], "info")
         # Log entire raw dictionary (as fallback)
        logger.info(f"BentoML [{task_name}] raw dict: {response}")
        # Log environment if error occurred
        if response.get("returncode", 0) != 0 or response.get("status") == "error":
            logger.error(f"BentoML [{task_name}] Fehler (Exitcode {response.get('returncode')})")
            logger.error(f"ENV: {os.environ}")
    else:
        log_multiline(response, "info")

def trigger_bento_training(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    url = "http://bentoml:4000/train_deep_hybrid_model"
    try:
        logging.info(f"Starte POST an {url} mit config: {conf}")
        resp = requests.post(url, json=conf, timeout=300)
        logging.info(f"BentoML-Training API Response: {resp.status_code} {resp.text}")

        try:
            resp_json = resp.json()
        except Exception as e:
            logging.error(f"Response nicht als JSON lesbar: {e}\nResponse-Text: {resp.text}")
            raise

        # Save backup log to file
        with open("/tmp/bento_airflow_response.json", "w") as f:
            json.dump(resp_json, f, indent=2)

        # log (incl. ENV Error)
        log_bento_response(resp_json, logging, "TRAIN")
        if resp.status_code != 200 or resp_json.get("status") == "error" or resp_json.get("returncode", 0) != 0:
            logging.error(f"Umgebungsvariablen bei Fehler: {os.environ}")
            raise Exception(f"BentoML Training Error: {resp_json}")
        return resp_json

    except requests.Timeout:
        logging.error("❗ BentoML-Training API Timeout nach 300 Sekunden!")
        logging.error(f"ENV bei Timeout: {os.environ}")
        raise
    except requests.ConnectionError as e:
        logging.error(f"❗ Netzwerkfehler beim Kontaktieren von BentoML: {e}")
        logging.error(f"ENV bei ConnectionError: {os.environ}")
        raise
    except Exception as e:
        logging.error(f"Fehler im Training-Call: {e}")
        logging.error(f"ENV bei Exception: {os.environ}")
        raise

def trigger_bento_validation(**context):
    conf = context["dag_run"].conf if "dag_run" in context and context["dag_run"] else {}
    url = "http://bentoml:4000/validate_model"
    val_conf = {"test_user_count": conf.get("test_user_count", 100)}
    try:
        resp = requests.post(url, json=val_conf)
        logging.info(f"BentoML-Validierung API Response: {resp.status_code} {resp.text}")
        resp_json = resp.json()
        log_bento_response(resp_json, logging, "VALIDATE")
        if resp.status_code != 200 or resp_json.get("status") == "error":
            raise Exception(f"BentoML Validation Error: {resp_json}")
        return resp_json
    except Exception as e:
        logging.error(f"Fehler im Validation-Call: {e}")
        raise

def trigger_bento_prediction(**context):
    url = "http://bentoml:4000/predict_best_model"
    try:
        resp = requests.post(url, json={})
        logging.info(f"BentoML-Prediction API Response: {resp.status_code} {resp.text}")
        resp_json = resp.json()
        log_bento_response(resp_json, logging, "PREDICT")
        if resp.status_code != 200 or resp_json.get("status") == "error":
            raise Exception(f"BentoML Prediction Error: {resp_json}")
        return resp_json
    except Exception as e:
        logging.error(f"Fehler im Prediction-Call: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}

with DAG(
    dag_id="bento_api_pipeline",
    default_args=default_args,
    description='Trigger Training/Validierung/Predict via BentoML API',
    schedule_interval=None,
    catchup=False,
) as dag:
    train_bento = PythonOperator(
        task_id="bento_train",
        python_callable=trigger_bento_training,
        provide_context=True
    )
    validate_bento = PythonOperator(
        task_id="bento_validate",
        python_callable=trigger_bento_validation,
        provide_context=True
    )
    predict_bento = PythonOperator(
        task_id="bento_predict",
        python_callable=trigger_bento_prediction
    )

    train_bento >> validate_bento >> predict_bento