# dags/movie_dag.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Korrekte Funktionen aus deinem wrappers.py importieren
from src.wrappers import (
    import_raw_data as preprocess_data,
    train_model,
    build_features as evaluate_model,
    predict_model as run_api
)

# Default DAG args
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2024, 1, 1),
}

# Define the DAG
with DAG(
    dag_id='movie_recommendation_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'movie'],
) as dag:

    t1 = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    t2 = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    t3 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    t4 = PythonOperator(
        task_id='run_api',
        python_callable=run_api
    )

    t1 >> t2 >> t3 >> t4