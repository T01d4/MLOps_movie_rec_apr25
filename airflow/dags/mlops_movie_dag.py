# === Airflow DAG (dags/mlops_movie_dag.py) ===
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.visualization.validate_predictions import validate_predictions
from src.visualization.predict_titles import main as predict_titles
from src.wrappers import (
    import_raw_data,
    make_dataset,
    build_features,
    train_model,
    predict_model
)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='mlops_movie_dag',
    default_args=default_args,
    schedule_interval=None,  # oder '* * * * *' für minütlich
    catchup=False
) as dag:

    task_import_data = PythonOperator(
        task_id='import_raw_data',
        python_callable=import_raw_data
    )

    task_make_dataset = PythonOperator(
        task_id='make_dataset',
        python_callable=make_dataset
    )

    task_build_features = PythonOperator(
        task_id='build_features',
        python_callable=build_features
    )

    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    task_predict_model = PythonOperator(
        task_id='predict_model',
        python_callable=predict_model
    )


    validate_predictions_task = PythonOperator(
        task_id="validate_predictions",
        python_callable=validate_predictions,
        dag=dag
    )

    task_predict_titles = PythonOperator(
        task_id="predict_titles",
        python_callable=predict_titles
    )

    task_import_data >> task_make_dataset >> task_build_features >> task_train_model >> task_predict_model >> validate_predictions_task >> task_predict_titles
 