import streamlit as st
import os
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
from requests.auth import HTTPBasicAuth
from auth import set_mlflow_from_env
from dotenv import load_dotenv
from recommender import show_best_model_info
import subprocess
import traceback

load_dotenv()

# === Für Airflow IMMER AIRFLOW_API_URL nehmen! ===
def fetch_dag_task_statuses():
    AIRFLOW_USER = "admin"
    AIRFLOW_PASS = "admin"
    airflow_url = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
    dag_id = "deep_models_pipeline"
    auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

    run_url = f"{airflow_url}/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"
    run_resp = requests.get(run_url, auth=auth)
    run_resp.raise_for_status()
    dag_run_id = run_resp.json()["dag_runs"][0]["dag_run_id"]

    task_url = f"{airflow_url}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    task_resp = requests.get(task_url, auth=auth)
    task_resp.raise_for_status()
    return task_resp.json()["task_instances"], dag_run_id

def fetch_airflow_logs(dag_id, dag_run_id, task_ids):
    logs = {}
    airflow_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1")
    auth = HTTPBasicAuth("admin", "admin")
    for task_id in task_ids:
        url = f"{airflow_url}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/1"
        resp = requests.get(url, auth=auth)
        if resp.ok:
            logs[task_id] = resp.text
        else:
            logs[task_id] = f"Fehler beim Abrufen des Logs: {resp.status_code}\n{resp.text}"
    return logs

def show_airflow_logs_tabbed(dag_id, dag_run_id, task_ids):
    logs = fetch_airflow_logs(dag_id, dag_run_id, task_ids)
    tabs = st.tabs(task_ids)
    for i, task_id in enumerate(task_ids):
        with tabs[i]:
            st.markdown(f"### Log: `{task_id}`")
            st.code(logs[task_id], language="log")


def formatalias(alias_str):
    """Formatiert die Aliase als Badges mit Emoji für @best_model."""
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
        # Aliase weiterhin versuchen (aber egal, da Workaround)
        if hasattr(v, "aliases") and v.aliases:
            alias_str = " | ".join(list(v.aliases))
        else:
            alias_str = ""
        tags = v.tags
        version = int(v.version)

        row = {
            "Version": int(v.version),
            "Created_at": pd.to_datetime(v.creation_timestamp, unit='ms').tz_localize('UTC').tz_convert('Europe/Berlin').strftime('%d.%m.%y %H:%M'),
            "Alias": alias_str,
            "precision_10": float(tags.get("precision_10", "nan")) if tags.get("precision_10") else float('nan'),
            "n_neighbors": tags.get("n_neighbors", ""),
            "latent_dim": tags.get("latent_dim", ""),
            "epochs": tags.get("epochs", ""),
            "tfidf_features": tags.get("tfidf_features", ""),
            "algorithm": tags.get("algorithm", ""),
            "tags": tags,  # Optional für Tooltip/Detail
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Keine Modelle in der Registry gefunden.")
        return
    
    # Markiere best_model-Version
    df["is_best"] = df["Version"] == best_model_version

    def highlight_best(row):
        if row["is_best"]:
            return '🏅 <b style="background:#ffd707;border-radius:6px;padding:2px 8px;color:#333">BEST</b>'
        return ""

    df["Alias"] = df.apply(lambda r: highlight_best(r), axis=1)

    df = df.sort_values("Version", ascending=False).reset_index(drop=True)
    df_display = df.drop(columns=["tags"])

    # Aliase als Badge/Emoji formatieren
    df_display["Alias"] = df_display["Alias"].apply(formatalias)

    # Precision als grün/rot
    def color_prec(val):
        if pd.isna(val):
            return ""
        color = "#27AE60" if val > 0.25 else "#E45756"
        return f'<b style="color:{color}">{val:.3f}</b>'
    df_display["precision_10"] = df_display["precision_10"].apply(color_prec)

    # Tabelle schön anzeigen
    st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Plotly: Linie mit Markierung "best_model"
    fig = go.Figure()
    best_idx = [i for i, a in enumerate(df["Alias"]) if "best_model" in a]
    fig.add_trace(go.Scatter(
        x=df["Created_at"],
        y=df["precision_10"].astype(float),
        mode="lines+markers",
        name="Precision@10",
        line=dict(color="#E45756", width=2),
        marker=dict(size=10, color=["#FFD700" if i in best_idx else "#E45756" for i in range(len(df))]),
        text=[
            "<br>".join([f"{k}: {v}" for k, v in tags.items()])
            for tags in df["tags"]
        ],
        hovertemplate=
            "Version: %{x}<br>Precision@10: %{y:.3f}<br>%{text}<extra></extra>"
    ))
    fig.update_layout(
        title="Precision@10 je Registry-Version",
        xaxis_title="Registrierungszeit",
        yaxis_title="Precision@10",
        height=400,
        plot_bgcolor="#f9fafb",
    )
    st.plotly_chart(fig, use_container_width=True)



def show_dag_progress():
    import time

    if st.session_state.get("dag_triggered", False):
        st.markdown("---")
        st.markdown("### 📋 Laufende DAG-Ausführung")

        progress_bar = st.progress(0)
        task_placeholder = st.empty()

        status_colors = {
            "success": "🟩",
            "failed": "🟥",
            "up_for_retry": "🟧",
            "running": "🔵",
            "queued": "⬜",
            "no_status": "⚪"
        }
        task_order = [
            "import_raw_data", "make_dataset", "build_features",
            "train_deep_hybrid_model", "validate_models", "predict_best_model"
        ]

        for attempt in range(40):
            try:
                tasks, run_id = fetch_dag_task_statuses()
            except Exception as e:
                st.error(f"Fehler beim Abrufen der Task-Status: {e}")
                break

            if tasks:
                task_states = {task["task_id"]: (task["state"] or "no_status") for task in tasks}
                task_output = f"🧭 Letzter Run: `{run_id}`\n\n"

                finished = sum(1 for t in task_order if task_states.get(t) == "success")
                total = len(task_order)
                percent = int((finished / total) * 100)
                progress_bar.progress(percent)

                for task_id in task_order:
                    status = task_states.get(task_id, "no_status")
                    emoji = status_colors.get(status, "⚪")
                    task_output += f"{emoji} `{task_id}` → **{status}**\n"

                task_placeholder.markdown(task_output)

                # === *** NEU: Zeige Logs von abgeschlossenen Tasks (Status success) ***
                finished_tasks = [t for t in task_order if task_states.get(t) == "success"]
                if finished_tasks:
                    st.markdown("#### ✅ Logs abgeschlossener Tasks")
                    tabs = st.tabs(finished_tasks)
                    logs = fetch_airflow_logs("deep_models_pipeline", run_id, finished_tasks)
                    for i, task_id in enumerate(finished_tasks):
                        with tabs[i]:
                            st.markdown(f"**Log für `{task_id}`**")
                            st.code(logs[task_id], language="log")
                # === ***

                if finished == total:
                    st.success("🎉 Alle Tasks abgeschlossen.")
                    st.session_state["dag_triggered"] = False
                    st.session_state["last_dag_run_id"] = run_id
                    break

            time.sleep(3)
            st.rerun()
            
    # **Log-Button und Anzeige nach Abschluss**
    if "last_dag_run_id" in st.session_state:
        if st.button("📜 Zeige Airflow Logs"):
            task_ids = [
                "import_raw_data", "make_dataset", "build_features",
                "train_deep_hybrid_model", "validate_models", "predict_best_model"
            ]
            st.subheader("Airflow Logs")
            show_airflow_logs_tabbed(
                dag_id="deep_models_pipeline",
                dag_run_id=st.session_state["last_dag_run_id"],
                task_ids=task_ids
            )

def show_admin_panel():
    st.header("👑 Admin Panel & Pipeline-Optionen")
    test_user_count = st.slider("Anzahl Test-User für Validierung", 10, 200, 100)
    n_neighbors = st.slider("KNN Nachbarn (n_neighbors)", 5, 50, 15)
    tfidf_features = st.slider("TF-IDF max_features", 50, 2000, 300)
    latent_dim = st.slider("Latente Dimension (latent_dim)", 8, 128, 32, step=8)
    epochs = st.slider("Epochen (epochs)", 5, 100, 30, step=5)

    st.session_state["pipeline_conf"] = {
        "test_user_count": test_user_count,
        "n_neighbors": n_neighbors,
        "tfidf_features": tfidf_features,
        "latent_dim": latent_dim,
        "epochs": epochs
    }


    AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
    AIRFLOW_USER = "admin"
    AIRFLOW_PASS = "admin"



    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Starte Deep Hybrid Training (DAG: deep_models_pipeline)"):
                try:
                    dag_id = "deep_models_pipeline"
                    airflow_url = AIRFLOW_API_URL  # <-- HIER KORRIGIERT!
                    auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

                    deep_conf = {
                        "test_user_count": test_user_count,
                        "n_neighbors": n_neighbors,
                        "latent_dim": latent_dim,
                        "tfidf_features": tfidf_features,
                        "epochs": epochs
                    }

                    # Step 1: Laufende Runs abbrechen
                    running_runs_url = f"{airflow_url}/dags/{dag_id}/dagRuns?state=running"
                    runs_resp = requests.get(running_runs_url, auth=auth)
                    runs_resp.raise_for_status()
                    running_runs = runs_resp.json().get("dag_runs", [])

                    for run in running_runs:
                        run_id = run["dag_run_id"]
                        patch_url = f"{airflow_url}/dags/{dag_id}/dagRuns/{run_id}"
                        patch_resp = requests.patch(patch_url, json={"state": "failed"}, auth=auth)
                        if patch_resp.status_code == 200:
                            st.info(f"❗ Abgebrochen: Run {run_id}")
                        else:
                            st.warning(f"⚠️ Konnte Run {run_id} nicht abbrechen: {patch_resp.text}")

                    # Step 2: Starte neuen Run
                    response = requests.post(
                        f"{airflow_url}/dags/{dag_id}/dagRuns",
                        json={"conf": deep_conf},
                        auth=auth
                    )
                    if response.status_code == 200:
                        st.success("✅ DAG Deep Hybrid Model wurde erfolgreich getriggert")
                        st.session_state["dag_triggered"] = True
                        st.session_state["progress"] = 0
                        st.session_state["last_dag"] = "deep_models_pipeline"
                    else:
                        st.error(f"❌ Fehler beim Triggern des DAGs: {response.text}")
                except Exception as e:
                    st.error(f"❌ Fehler bei API-Aufruf: {e}")
  

    with col2:
        with st.expander("📊 Zeige Registry-Modelle & Tags (DagsHub)", expanded=False):
            try:
                show_registry_metrics()
            except Exception as e:
                st.error(f"Fehler beim Laden der Registry-Metriken: {e}")
    show_dag_progress()