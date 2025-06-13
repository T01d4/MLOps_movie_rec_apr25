# === api_service/main.py ===

from fastapi import FastAPI, UploadFile, HTTPException, Depends, File, Query, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from trainer import router as trainer_router
from recommend import router as recommend_router  # <--  router with /recommend!
import pandas as pd
import numpy as np
import mlflow
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import requests
import os
import json
from pathlib import Path


load_dotenv(".env")
app = FastAPI()
app.include_router(trainer_router)
app.include_router(recommend_router) 

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT = "hybrid_deep_model"  # or the name used during training


SECRET_KEY = os.getenv("SECRET_KEY", "supersecret123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

USERS_FILE = Path(__file__).parent / "users.json"
with open(USERS_FILE, "r", encoding="utf-8") as f:
    fake_users_db = json.load(f)
    
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    return db.get(username)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"]}




@app.post("/train")
def train_model():
    
    airflow_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080") + "/api/v1/dags/deep_models_pipeline/dagRuns"

    response = requests.post(
        airflow_url,
        auth=("admin", "admin"),
        json={"conf": {}}   # empty, everything is read internally
    )

    if response.status_code in (200, 201):
        data = response.json()
        run_id = data.get("dag_run_id") or data.get("run_id")
        return {"status": "Train DAG triggered", "dag_run_id": run_id}
    else:
        return {"status": "Error", "code": response.status_code, "response": response.text}

@app.post("/validate")
def validate_model(run_id: str = Body(...)):
    # Starts validation via Airflow (can pass run_id as config!)
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
    # Load based on passed run_id, version or alias
    model_uri = None
    if run_id:
        # Look up model version for run_id (via MLflow)
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
    # Continue as usual: read file, load model, predict

@app.get("/metrics")
def get_metrics(run_id: str = Query(None), model_alias: str = Query("best_model")):
    # Metric output for a specific run or the best_model
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
