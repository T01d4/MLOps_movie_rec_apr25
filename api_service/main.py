# === api_service/main.py ===

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
import mlflow
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import requests
import os

load_dotenv(".env")
app = FastAPI()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT = "hybrid_deep_model"  # oder wie im Training

@app.post("/train")
def train_model(n_neighbors: int = 10, latent_dim: int = 64, epochs: int = 30, tfidf_features: int = 300):
    # Starte Airflow DAG über REST (wie gehabt)
    airflow_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080") + "/api/v1/dags/deep_models_pipeline/dagRuns"
    conf = {
        "n_neighbors": n_neighbors,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "tfidf_features": tfidf_features,
    }
    response = requests.post(
        airflow_url,
        auth=("admin", "admin"),
        json={"conf": conf}
    )
    if response.status_code in (200, 201):
        data = response.json()
        run_id = data.get("dag_run_id") or data.get("run_id")
        # <- Das in st.session_state speichern!
        return {"status": "Train DAG triggered", "dag_run_id": run_id, "conf": conf}

@app.post("/validate")
def validate_model(run_id: str = Body(...)):
    # Startet Validierung per Airflow (kann als conf die run_id weitergeben!)
    airflow_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080") + "/api/v1/dags/deep_models_pipeline/dagRuns"
    response = requests.post(
        airflow_url,
        auth=("admin", "admin"),
        json={"conf": {"run_id": run_id, "step": "validate"}}
    )
    if response.status_code in (200, 201):
        return {"status": "Validate DAG triggered", "run_id": run_id}
    return JSONResponse(status_code=response.status_code, content={"error": response.text})

@app.post("/predict")
async def predict(file: UploadFile = File(...), run_id: str = Query(None), model_version: int = Query(None), model_alias: str = Query("best_model")):
    # Laden je nach übergebenem run_id, version oder alias
    model_uri = None
    if run_id:
        # Suche Modellversion zu Run-ID (geht über mlflow)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MLFLOW_EXPERIMENT}'")
        mv = [v for v in versions if v.run_id == run_id]
        if mv:
            model_uri = f"models:/{MLFLOW_EXPERIMENT}/{mv[0].version}"
    elif model_version:
        model_uri = f"models:/{MLFLOW_EXPERIMENT}/{model_version}"
    else:
        model_uri = f"models:/{MLFLOW_EXPERIMENT}@{model_alias}"

    if not model_uri:
        return JSONResponse(status_code=400, content={"error": "Kein Modell angegeben"})
    # Dann wie gehabt: Datei lesen, Modell laden, predicten

@app.get("/metrics")
def get_metrics(run_id: str = Query(None), model_alias: str = Query("best_model")):
    # Metrik-Output für einen bestimmten Run oder das best_model
    client = mlflow.tracking.MlflowClient()
    if run_id:
        run = mlflow.get_run(run_id)
        return run.data.metrics
    else:
        versions = client.search_model_versions(f"name='{MLFLOW_EXPERIMENT}'")
        best = None
        for v in versions:
            if model_alias in v.aliases:
                best = v
                break
        if not best:
            return JSONResponse(status_code=404, content={"error": f"Alias {model_alias} not found"})
        run = mlflow.get_run(best.run_id)
        return run.data.metrics

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080")
USERNAME = os.getenv("AIRFLOW_USER", "admin")
PASSWORD = os.getenv("AIRFLOW_PW", "admin")


@app.get("/dag/last_run_status")
def get_last_dag_run_status(dag_id: str = Query(...)):
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200:
        dag_runs = response.json().get("dag_runs", [])
        if dag_runs:
            return {"dag_run_id": dag_runs[0]["dag_run_id"], "state": dag_runs[0]["state"]}
        else:
            return {"error": "No runs found."}
    return {"error": response.text}

@app.get("/dag/tasks")
def get_task_statuses(dag_id: str = Query(...), dag_run_id: str = Query(...)):
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200:
        return {"task_instances": response.json().get("task_instances", [])}
    return {"error": response.text}