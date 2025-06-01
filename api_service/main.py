# === api_service/main.py ===

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import os
import mlflow
from dotenv import load_dotenv
from predict import predict_from_csv
import requests
import pickle
from requests.auth import HTTPBasicAuth

load_dotenv(".env")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Movie Recommender API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        preds = predict_from_csv(temp_path)
        df = pd.DataFrame(preds)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_path)

@app.post("/train")
def train():
    airflow_url = "http://airflow-webserver:8080/api/v1/dags/movie_recommendation_pipeline/dagRuns"
    try:
        response = requests.post(
            airflow_url,
            headers={"Authorization": "Basic YWlyZmxvdzphaXJmbG93"},  # admin:admin
            json={"conf": {}, "dag_run_id": "manual_trigger"}
        )
        if response.status_code == 200:
            return {"status": "Training DAG triggered successfully."}
        else:
            return JSONResponse(status_code=response.status_code, content={"error": response.text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def metrics():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    experiment = mlflow.get_experiment_by_name("movie_recommendation")
    if not experiment:
        return JSONResponse(status_code=404, content={"error": "Experiment not found."})

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    best_run = runs.sort_values("metrics.map_10", ascending=False).iloc[0]

    # Speichere feature_names.txt aus Modellverzeichnis (falls vorhanden)
    try:
        feature_path = "/app/models/feature_names.txt"
        columns_path = "/opt/airflow/model_cache/columns.pkl"
        if os.path.exists(columns_path):
            with open(columns_path, "rb") as f_in:
                feature_names = pickle.load(f_in)
            os.makedirs("/app/models", exist_ok=True)
            with open(feature_path, "w") as f_out:
                f_out.write("\n".join(feature_names))
    except Exception as e:
        print(f"⚠️ Fehler beim Schreiben von feature_names.txt: {e}")

    return {
        "run_id": best_run.run_id,
        "map_10": best_run.get("metrics.map_10", "-"),
        "precision": best_run.get("metrics.precision_10", "-"),
        "recall": best_run.get("metrics.recall", "-"),
        "rmse": best_run.get("metrics.rmse", "-")
    }

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
DAG_ID = "movie_recommendation_pipeline"
USERNAME = os.getenv("AIRFLOW_USER", "admin")
PASSWORD = os.getenv("AIRFLOW_PW", "admin")

def get_dag_run_status(dag_id):
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200:
        return response.json()["dag_runs"][0]["dag_run_id"]
    return None

def get_task_statuses(dag_id, dag_run_id):
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200:
        return response.json()["task_instances"]
    return []