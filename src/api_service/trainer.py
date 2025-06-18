# api_service/trainer.py
import os
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
import requests
import time
import html
from pydantic import BaseModel, Field

load_dotenv()

router = APIRouter()

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")
AUTH = (AIRFLOW_USER, AIRFLOW_PASS)

DAGS = {
    "deep_models_pipeline": {
        "label": "Deep Hybrid Training",
        "task_order": [
            "import_raw_data", "make_dataset", "build_features",
            "train_deep_hybrid_model", "validate_models", "predict_best_model",
            "trigger_drift_monitoring_dag"
        ]
    },
    "bento_api_pipeline": {
        "label": "BentoML-Pipeline",
        "task_order": ["bento_train", "bento_validate", "bento_predict"]
    },
    "drift_monitoring_dag": {
        "label": "Drift Monitoring",
        "task_order": [
            "generate_embedding_snapshot",
            "analyze_snapshot_drift",
            "analyze_request_drift",
            "generate_drift_report_extended"
        ]
    }
}

@router.get("/airflow/dag-metadata")
def get_dag_metadata():
    return DAGS

@router.get("/airflow/dag-status")
def get_dag_status(dag_id: str):
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}"
    try:
        resp = requests.get(url, auth=AUTH, timeout=5)
        resp.raise_for_status()
        return {"active": not resp.json()["is_paused"]}
    except Exception as e:
        return {"error": f"Error while retrieving status for {dag_id}: {str(e)}"}

@router.post("/airflow/set-dag-status")
def set_dag_status(data: dict = Body(...)):
    dag_id = data["dag_id"]
    enable = data["enable"]
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}"
    try:
        resp = requests.patch(url, json={"is_paused": not enable}, auth=AUTH)
        resp.raise_for_status()
        return {"ok": resp.ok}
    except Exception as e:
        return {"error": f"Error while toggling the DAG: {str(e)}"}

@router.post("/airflow/trigger-dag")
def trigger_dag(dag_id: str, conf: dict = Body(...)):
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
    try:
        resp = requests.post(url, json={"conf": conf}, auth=AUTH)
        return {"status_code": resp.status_code, "content": resp.json()}
    except Exception as e:
        return {"error": str(e)}

@router.get("/airflow/last-run-id")
def fetch_last_run_id(dag_id: str):
    run_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"
    run_resp = requests.get(run_url, auth=AUTH)
    run_resp.raise_for_status()
    dag_runs = run_resp.json().get("dag_runs", [])
    if not dag_runs:
        return None
    return {"dag_run_id": dag_runs[0]["dag_run_id"]}

@router.get("/airflow/task-statuses")
def fetch_task_statuses(dag_id: str):
    run_id_resp = fetch_last_run_id(dag_id)
    dag_run_id = run_id_resp["dag_run_id"] if run_id_resp else None
    if not dag_run_id:
        return {"tasks": None, "dag_run_id": None}
    task_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    resp = requests.get(task_url, auth=AUTH)
    resp.raise_for_status()
    return {"tasks": resp.json()["task_instances"], "dag_run_id": dag_run_id}

@router.get("/airflow/logs")
def fetch_airflow_logs(dag_id: str, dag_run_id: str):
    task_order = DAGS[dag_id]["task_order"]
    logs = {}
    for task_id in task_order:
        url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/1"
        resp = requests.get(url, auth=AUTH)
        logs[task_id] = resp.text if resp.ok else f"Error while fetching logs: {resp.status_code}\n{resp.text}"
    return logs

@router.post("/airflow/abort-runs")
def abort_running_dag_runs(dag_id: str):
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns?state=running"
    resp = requests.get(url, auth=AUTH)
    if not resp.ok:
        return {"error": "Could not fetch DAG runs"}
    for run in resp.json().get("dag_runs", []):
        patch_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{run['dag_run_id']}"
        requests.patch(patch_url, json={"state": "failed"}, auth=AUTH)
    return {"aborted": True}

@router.get("/airflow/progress")
def show_dag_progress(dag_id: str):
    task_order = DAGS[dag_id]["task_order"]
    statuses = fetch_task_statuses(dag_id)
    tasks = statuses.get("tasks", [])
    run_id = statuses.get("dag_run_id")
    if not tasks or not run_id:
        return {"progress": []}
    task_states = {task["task_id"]: (task["state"] or "no_status") for task in tasks}
    finished = sum(1 for t in task_order if task_states.get(t) == "success")
    total = len(task_order)
    percent = int((finished / total) * 100)
    task_output = f"🧭 Letzter Run: `{run_id}`\n\n"
    for task_id in task_order:
        status = task_states.get(task_id, "no_status")
        emoji = "🟩" if status == "success" else (
            "🟥" if status == "failed" else (
                "🟧" if status == "up_for_retry" else (
                    "🔵" if status == "running" else (
                        "⬜" if status == "queued" else "⚪"
                    )
                )
            )
        )
        task_output += f"{emoji} `{task_id}` → **{status}**\n"
    logs = fetch_airflow_logs(dag_id, run_id)
    step = {
        "percent": percent,
        "task_output": task_output,
        "logs": logs,
        "finished": finished == total
    }
    #  Drift-DAG successfull
    drift_triggered = (
        dag_id == "deep_models_pipeline"
        and task_states.get("trigger_drift_monitoring_dag") == "success"
    )
    step["triggered_dag_success"] = drift_triggered
    return {"progress": [step]}

def formatalias(alias_str):
    if not alias_str:
        return ""
    badges = []
    for a in alias_str.split(" | "):
        if not a:
            continue
        if a == "best_model" or "@best_model" in a:
            badges.append(
                '🏅 <span style="background:#ffd707;border-radius:6px;padding:2px 8px;color:#333;font-weight:600">@best_model</span>'
            )
        else:
            badges.append(
                f'<span style="background:#eee;border-radius:6px;padding:2px 8px;color:#333;font-weight:600">@{a}</span>'
            )
    return " ".join(badges)

@router.get("/mlflow/registry-metrics")
def show_registry_metrics():
    model_name = "hybrid_deep_model"
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    rows = []

    try:
        mv = client.get_model_version_by_alias(model_name, "best_model")
        best_model_version = int(mv.version)
    except Exception:
        best_model_version = None

    for v in versions:
        if hasattr(v, "aliases") and v.aliases:
            alias_str = " | ".join(list(v.aliases))
        else:
            alias_str = ""
        tags = v.tags
        version = int(v.version)
        row = {
            "Version": int(v.version),
            "Created_at": pd.to_datetime(v.creation_timestamp, unit='ms')
                            .tz_localize('UTC')
                            .tz_convert('Europe/Berlin')
                            .strftime('%d.%m.%y %H:%M'),
            "Alias": alias_str,
            "precision_10": float(tags.get("precision_10", "nan")) if tags.get("precision_10") else float('nan'),

            # Training parameters
            "n_neighbors": tags.get("n_neighbors", ""),
            "latent_dim": tags.get("latent_dim", ""),
            "hidden_dim": tags.get("hidden_dim", ""),
            "epochs": tags.get("epochs", ""),
            "lr": tags.get("lr", ""),
            "batch_size": tags.get("batch_size", ""),
            "tfidf_features": tags.get("tfidf_features", ""),
            "metric": tags.get("metric", ""),        
            "content_weight": tags.get("content_weight", ""),
            "collab_weight": tags.get("collab_weight", ""),
            "power_factor": tags.get("power_factor", ""),
            "drop_threshold": tags.get("drop_threshold", ""),
            "tags": tags,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"message": "Keine Modelle in der Registry gefunden."}

    df["is_best"] = df["Version"] == best_model_version

    def highlight_best(row):
        if row["is_best"]:
            return '🏅 <b style="background:#ffd707;border-radius:6px;padding:2px 8px;color:#333">BEST</b>'
        return ""

    df["Alias"] = df.apply(lambda r: highlight_best(r), axis=1)
    df = df.sort_values("Version", ascending=False).reset_index(drop=True)
    df_display = df.drop(columns=["tags"])
    df_display["Alias"] = df_display["Alias"].apply(formatalias)

    def color_prec(val):
        if pd.isna(val):
            return ""
        color = "#27AE60" if val > 0.25 else "#E45756"
        return f'<b style="color:{color}">{val:.3f}</b>'
    df_display["precision_10"] = df_display["precision_10"].apply(color_prec)

    html_table = df_display.to_html(escape=False, index=False)

    plot_data = {
        "x": df["Created_at"].tolist(),
        "y": df["precision_10"].astype(float).tolist(),
        "marker_color": ["#FFD700" if r else "#E45756" for r in df["is_best"]],
        "text": [
            "<br>".join([f"{k}: {v}" for k, v in tags.items()])
            for tags in df["tags"]
        ]
    }

    return {
        "table_html": html_table,
        "plot_data": plot_data,
        "raw_rows": df_display.to_dict(orient="records"),
    }