#training.py
import streamlit as st
import os
import requests
import pandas as pd
import plotly.graph_objects as go
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import time
import json


load_dotenv()

API_URL = os.getenv("API_URL", "http://api_service:8000")
DAGS = {
    "deep_models_pipeline": {
        "label": "Deep Hybrid Training",
        "task_order": [
            "import_raw_data", "make_dataset", "build_features",
            "train_deep_hybrid_model", "validate_models", "predict_best_model"
        ]
    },
    "bento_api_pipeline": {
        "label": "BentoML-Pipeline",
        "task_order": ["bento_train", "bento_validate", "bento_predict"]
    }
}

def get_dag_status(dag_id):
    url = f"{API_URL}/airflow/dag-status?dag_id={dag_id}"
    try:
        resp = requests.get(url, timeout=5)
        return resp.json().get("active")
    except Exception as e:
        st.warning(f"Fehler beim Abrufen des Status für {dag_id}: {e}")
        return None

def set_dag_status(dag_id, enable):
    url = f"{API_URL}/airflow/set-dag-status"
    try:
        resp = requests.post(url, json={"dag_id": dag_id, "enable": enable})
        return resp.json().get("ok")
    except Exception as e:
        st.error(f"Fehler beim Umschalten des DAGs: {e}")
        return False

def trigger_dag(dag_id, conf):
    url = f"{API_URL}/airflow/trigger-dag?dag_id={dag_id}"
    resp = requests.post(url, json={"conf": conf})
    return resp

def show_dag_progress(dag_id):
    task_order = DAGS[dag_id]["task_order"]
    progress_placeholder = st.empty()
    task_placeholder = st.empty()
    log_placeholder = st.empty()
    active = True
    last_percent = -1

    while active:
        try:
            progress_data = requests.get(f"{API_URL}/airflow/progress?dag_id={dag_id}").json()
            steps = progress_data.get("progress", [])
            if not steps:
                task_placeholder.info("Noch kein aktiver Run.")
                break

            step = steps[-1]
            percent = step.get("percent", 0)
            finished = step.get("finished", False)

            # Fortschrittsbalken nur aktualisieren, wenn sich was ändert (sonst flackert er!)
            if percent != last_percent:
                progress_placeholder.progress(percent)
                last_percent = percent

            task_placeholder.markdown(step["task_output"])
            log_placeholder.markdown("#### 📝 Logs")
            logs = step["logs"]
            tabs = log_placeholder.tabs(task_order)
            for i, task_id in enumerate(task_order):
                with tabs[i]:
                    st.markdown(f"**Task:** `{task_id}`")
                    st.code(logs.get(task_id, "Kein Log gefunden."), language="log")

            if finished:
                progress_placeholder.progress(100)
                st.success("🎉 Alle Tasks abgeschlossen.")
                active = False
            else:
                # 1 Sekunde warten, dann neu abfragen
                time.sleep(1)
        except Exception as e:
            st.error(f"Fehler beim Polling: {e}")
            break

def poll_and_rerun(dag_id, min_interval=2.0):
    # Neues Pattern für flüssiges Polling!
    now = time.time()
    last_poll_key = f"{dag_id}_last_poll"
    last_poll = st.session_state.get(last_poll_key, 0)
    finished = show_dag_progress(dag_id)
    if not finished:
        # Warte minimal min_interval Sekunden, bevor neu geladen wird
        if now - last_poll > min_interval:
            st.session_state[last_poll_key] = now
            st.experimental_rerun()

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

def show_registry_metrics():
    url = f"{API_URL}/mlflow/registry-metrics"
    resp = requests.get(url)
    j = resp.json()
    if "table_html" in j:
        st.markdown(j["table_html"], unsafe_allow_html=True)
    else:
        st.info(j.get("message", "Keine Modelle in der Registry gefunden."))
    plot_data = j.get("plot_data", {})
    if plot_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data["x"],
            y=plot_data["y"],
            mode="lines+markers",
            name="Precision@10",
            line=dict(color="#E45756", width=2),
            marker=dict(size=10, color=plot_data["marker_color"]),
            text=plot_data["text"],
            hovertemplate="Version: %{x}<br>Precision@10: %{y:.3f}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            title="Precision@10 je Registry-Version",
            xaxis_title="Registrierungszeit",
            yaxis_title="Precision@10",
            height=400,
            plot_bgcolor="#f9fafb",
        )
        st.plotly_chart(fig, use_container_width=True)

def dag_toggle_ui(dag_id):
    status = get_dag_status(dag_id)
    label = DAGS[dag_id]["label"]
    st.subheader(f"🟢 / 🔴 Airflow Pipeline steuern: **{label}**")
    if status is None:
        st.warning("Konnte Status nicht abfragen.")
        return
    if status:
        st.success(f"🟢 {label} ist **AKTIV**")
        if st.button(f"🛑 Deaktiviere {label}"):
            if set_dag_status(dag_id, False):
                st.info(f"{label} wurde deaktiviert.")
                st.experimental_rerun()
    else:
        st.warning(f"🔴 {label} ist **INAKTIV**")
        if st.button(f"✅ Aktiviere {label}"):
            if set_dag_status(dag_id, True):
                st.success(f"{label} wurde aktiviert.")
                st.experimental_rerun()

def show_admin_panel():
    st.header("👑 Admin Panel & Pipeline-Optionen")

    # Pipeline-Parameter
    test_user_count = st.slider("Anzahl Test-User für Validierung", 10, 200, 100)
   # Hauptparameter (einspaltig)
    n_neighbors = st.slider("KNN neighbors (n_neighbors)", 5, 80, 10)
    latent_dim = st.slider("Latent dimension (latent_dim)", 8, 128, 32, step=8)
    hidden_dim = st.slider("Hidden layer size (hidden_dim)", 16, 512, 256, step=16)
    tfidf_features = st.slider("TF-IDF max_features (tfidf_features)", 50, 3000, 300)
    epochs = st.slider("Epochs", 5, 100, 30, step=5)
    lr = st.select_slider("Learning rate (lr)", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128, 256], index=3)
    metric = st.selectbox("Distance metric", ["cosine", "euclidean", "manhattan"], index=0)

    # Gewichte und Schwellenwerte kompakter in 2 Spalten
    col1, col2 = st.columns(2)
    with col1:
        content_weight = st.slider("Content Weight", 0.0, 1.0, 0.5, step=0.1)
        power_factor = st.slider("Power Factor", 0.1, 5.0, 1.0, step=0.1)
    with col2:
        collab_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.5, step=0.1)
        drop_threshold = st.slider("Drop Threshold", 0.0, 1.0, 0.0, step=0.05)

    # Speichern
    st.session_state["pipeline_conf"] = {"test_user_count": test_user_count}
    config_dict = {
        "n_neighbors": n_neighbors,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "tfidf_features": tfidf_features,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "metric": metric,
        "content_weight": content_weight,
        "collab_weight": collab_weight,
        "power_factor": power_factor,
        "drop_threshold": drop_threshold
    }

    data_dir = os.getenv("DATA_DIR", "/app/data")
    config_path = os.path.join(data_dir, "processed", "pipeline_conf.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    st.success("✅ Pipeline-Konfiguration gespeichert!")

    # 🔁 DAG Trigger-UI (volle Breite)
    for dag_id in DAGS:
        label = DAGS[dag_id]["label"]
        dag_toggle_ui(dag_id)
        if get_dag_status(dag_id):
            btn_label = f"▶️ Starte {label} (DAG: {dag_id})"
            if st.button(btn_label, key=f"run_{dag_id}"):
                AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
                AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
                AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")
                auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

                running_runs_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns?state=running"
                runs_resp = requests.get(running_runs_url, auth=auth)
                if runs_resp.ok:
                    for run in runs_resp.json().get("dag_runs", []):
                        run_id = run["dag_run_id"]
                        patch_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{run_id}"
                        requests.patch(patch_url, json={"state": "failed"}, auth=auth)

                response = trigger_dag(dag_id, st.session_state["pipeline_conf"])
                if response.status_code in (200, 201):
                    st.success(f"{label} gestartet!")
                    st.session_state[f"{dag_id}_triggered"] = True
                else:
                    st.error(response.text)
        else:
            st.info(f"ℹ️ {label} ist **deaktiviert** – aktiviere ihn oben, um zu starten.")

        if st.session_state.get(f"{dag_id}_triggered", False):
            show_dag_progress(dag_id)
            st.session_state[f"{dag_id}_triggered"] = False

    # 🔧 BentoML Service UI (volle Breite)
    with st.expander("🛠️ BentoML Service (Docker)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ BentoML Service STARTEN", key="bento_start"):
                os.system("docker compose up -d bentoml_service")
                st.success("BentoML-Service gestartet!")
                st.experimental_rerun()
        with col2:
            if st.button("🛑 BentoML Service STOPPEN", key="bento_stop"):
                os.system("docker compose stop bentoml_service")
                st.warning("BentoML-Service gestoppt!")
                st.experimental_rerun()

        def is_bento_running():
            import subprocess
            result = subprocess.run(
                "docker ps --filter 'name=bentoml_service' --filter 'status=running' --format '{{.Names}}'",
                shell=True, capture_output=True, text=True
            )
            return "bentoml_service" in result.stdout

        if is_bento_running():
            st.success("🟢 BentoML-Service läuft")
        else:
            st.error("🔴 BentoML-Service gestoppt (manuell oder automatisch).")

    # 📊 MLflow Registry Metrics (volle Breite)
    with st.expander("📊 Zeige Registry-Modelle & Tags (DagsHub)", expanded=False):
        try:
            show_registry_metrics()
        except Exception as e:
            st.error(f"Fehler beim Laden der Registry-Metriken: {e}")