# mlops_movie_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.visualization.validate_predictions import validate_predictions
from src.visualization.predict_titles import main as predict_titles
from src.wrappers import (
    import_raw_data,
    make_dataset,
    build_features,
    train_model
)

# ✅ Funktion zum Erzeugen der Eingabedaten
def generate_input_data():
    import pandas as pd
    import os

    input_path = "/opt/airflow/data/input.csv"
    matrix_path = "/opt/airflow/data/processed/movies_matrix.csv"

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"❌ movies_matrix.csv nicht gefunden unter {matrix_path}")

    df = pd.read_csv(matrix_path)

    # Nutze die Mittelwerte der ersten 3 Filme
    df_sample = df.head(3).mean(numeric_only=True).to_frame().T
    df_sample.to_csv(input_path, index=False)
    print(f"✅ Neue Eingabe gespeichert unter: {input_path}")

# === Neue Funktion: Mapping-Prediction mit movieId
def run_prediction():
    import os
    import pandas as pd
    import pickle
    from dotenv import load_dotenv
    from src.visualization.validate_predictions import validate_predictions

    load_dotenv(dotenv_path="/opt/airflow/.env")

    input_path = "/opt/airflow/data/input.csv"
    output_dir = "/opt/airflow/data/predictions"
    output_path = os.path.join(output_dir, "predicted_titles.csv")
    prediction_matrix = os.path.join(output_dir, "predictions.csv")

    model_path = "/opt/airflow/models/model.pkl"
    columns_path = "/opt/airflow/model_cache/columns.pkl"
    movie_ids_path = "/opt/airflow/models/model_ids.pkl"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Eingabedatei fehlt: {input_path}")

    df = pd.read_csv(input_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(columns_path, "rb") as f:
        expected_columns = pickle.load(f)
    df = df[expected_columns]

    _, indices = model.kneighbors(df)

    with open(movie_ids_path, "rb") as f:
        movie_ids = pickle.load(f)
    predictions = [[movie_ids[i] for i in row] for row in indices]

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(prediction_matrix, index=False, header=False)

    # ✅ Fix: Spalten dynamisch benennen
    top_n = len(predictions[0])
    col_names = [f"title_{i+1}" for i in range(top_n)]
    pd.DataFrame([predictions[0]], columns=col_names).to_csv(output_path, index=False)

    validate_predictions(output_path)

# === DAG-Konfiguration ===
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='mlops_movie_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Trainiert Modell, erzeugt Eingabe, validiert Vorhersagen, visualisiert"
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
        python_callable=run_prediction
    )

    task_generate_input = PythonOperator(
        task_id='generate_input_data',
        python_callable=generate_input_data
    )

    task_validate_predictions = PythonOperator(
        task_id="validate_predictions",
        python_callable=validate_predictions,
        op_args=["/opt/airflow/data/predictions/predicted_titles.csv"]
    )

    task_predict_titles = PythonOperator(
        task_id="predict_titles",
        python_callable=predict_titles,
        op_kwargs={
            "pred_path": "/opt/airflow/data/predictions/predictions.csv",
            "movies_path": "/opt/airflow/data/raw/movies.csv",
            "output_dir": "/opt/airflow/data/predictions",
            "mapping_path": "/opt/airflow/models/model_ids.pkl"
        }
    )

    # === DAG-Abfolge ===
    task_import_data >> task_make_dataset >> task_build_features >> task_train_model >> task_predict_model
    task_predict_model >> task_generate_input >> task_validate_predictions >> task_predict_titles