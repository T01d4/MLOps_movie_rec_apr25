#streamlit/training.py
import streamlit as st
import os
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px 
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import time
import json
import streamlit.components.v1 as components
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from src.monitoring.plot_precision_history import load_precision_history, plot_and_save_precision, save_metrics_csv
import subprocess

load_dotenv()

API_URL = os.getenv("API_URL", "http://api_service:8000")

def fetch_dag_metadata():
    try:
        return requests.get(f"{API_URL}/airflow/dag-metadata").json()
    except Exception as e:
        st.warning(f"⚠️ Cannot fetch DAG metadata: {e}")
        return {}

DAGS = fetch_dag_metadata()

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
REPORT_DIR = os.getenv("REPORT_DIR", "/app/reports")
MONITOR_DIR = os.path.join(DATA_DIR, "monitoring")
METRICS_PATH = os.path.join(MONITOR_DIR, "metrics_from_mlflow.csv")
GRAFANA_URL = os.getenv("GRAFANA_URL", "")
drift_html_path = os.path.join(REPORT_DIR, "drift_report.html")
drift_html_pathext = os.path.join(REPORT_DIR, "drift_report_extended.html")


def get_dag_status(dag_id):
    url = f"{API_URL}/airflow/dag-status?dag_id={dag_id}"
    try:
        resp = requests.get(url, timeout=5)
        return resp.json().get("active")
    except Exception as e:
        st.warning(f"⚠️ Failed to retrieve DAG status for {dag_id}: {e}")
        return None

def set_dag_status(dag_id, enable):
    url = f"{API_URL}/airflow/set-dag-status"
    try:
        resp = requests.post(url, json={"dag_id": dag_id, "enable": enable})
        return resp.json().get("ok")
    except Exception as e:
        st.error(f"❌ Failed to toggle DAG status: {e}")
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
    drift_triggered = False  # Flag for showing monitoring later

    while active:
        try:
            progress_data = requests.get(f"{API_URL}/airflow/progress?dag_id={dag_id}").json()
            steps = progress_data.get("progress", [])
            if not steps:
                task_placeholder.info("No active run yet.")
                break

            step = steps[-1]
            percent = step.get("percent", 0)
            finished = step.get("finished", False)
            drift_triggered = step.get("triggered_dag_success", False)

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
                    st.code(logs.get(task_id, "No log found."), language="log")

            if finished:
                progress_placeholder.progress(100)
                st.success("🎉 All tasks completed.")
                active = False
            else:
                time.sleep(1)

        except Exception as e:
            st.error(f"Polling error: {e}")
            break

    # After the main loop – show monitoring logs once
    if drift_triggered:
        st.markdown("---")
        st.subheader("🧪 Logs: Drift Monitoring Pipeline")
        show_dag_progress("drift_monitoring_dag")

def poll_and_rerun(dag_id, min_interval=2.0):
    # new pattern for Polling!
    now = time.time()
    last_poll_key = f"{dag_id}_last_poll"
    last_poll = st.session_state.get(last_poll_key, 0)
    finished = show_dag_progress(dag_id)
    if not finished:
        # wait min_interval Second before reloading
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
        st.info(j.get("message", "No models found in registry."))
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
    st.subheader(f"🟢 / 🔴 Airflow Pipeline Control: **{label}**")
    if status is None:
        st.warning("⚠️ Could not retrieve DAG status.")
        return
    if status:
        st.success(f"🟢 {label} ist **AKTIV**")
        if st.button(f"🛑 Deaktiviere {label}"):
            if set_dag_status(dag_id, False):
                st.info(f"{label} has been deactivated.")
                st.experimental_rerun()
    else:
        st.warning(f"🔴 {label} is **INACTIVE**")
        if st.button(f"✅ Aktiviere {label}"):
            if set_dag_status(dag_id, True):
                st.success(f"{label} has been activated.")
                st.experimental_rerun()


def show_drift_score():
    st.subheader("🧪 Monitoring & Drift Detection")

    drift_path = os.path.join(REPORT_DIR, "drift_metrics.json")
    if not os.path.exists(drift_path):
        st.warning("⚠️ No drift_metrics.json found.")
        return

    with open(drift_path, "r") as f:
        drift = json.load(f)

    drift_score = None
    for m in drift.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            drift_score = m.get("result", {}).get("drift_share", None)
            break

    if drift_score is not None:
        color = "green" if drift_score < 0.3 else "orange" if drift_score < 0.6 else "red"
        st.markdown(f"### Gesamtdrift: <span style='color:{color}'><b>{drift_score:.2f}</b></span>", unsafe_allow_html=True)
        if drift_score < 0.3:
            st.success("✅ No significant drift detected.")
        elif drift_score < 0.6:
            st.warning("⚠️ Slight drift detected.")
        else:
            st.error("🚨 Strong drift! Re-training recommended.")
    else:
        st.info("ℹ️ No drift score found.")

def show_grafana_dashboard():
    grafana_url = os.getenv("GRAFANA_URL", "")
    if grafana_url:
        components.iframe(grafana_url, height=800, scrolling=True)
    else:
        st.info("ℹ️ No Grafana URL set.")

def show_monitoring_downloads():
    files_found = False
    for file in os.listdir(REPORT_DIR):
        if file.endswith((".json", ".png", ".html")):
            with open(os.path.join(REPORT_DIR, file), "rb") as f:
                st.download_button(f"⬇️ {file}", f, file_name=file)
                files_found = True
    if not files_found:
        st.info("ℹ️ No monitoring files found.")

def show_model_comparison_charts():
    data_dir = os.getenv("DATA_DIR", "/app/data")
    csv_path = os.path.join(data_dir, "monitoring", "metrics_from_mlflow.csv")

    if not os.path.exists(csv_path):
        st.warning("⚠️ metrics_from_mlflow.csv not found.")
        return

    df = pd.read_csv(csv_path)

    if df.empty or "precision_10" not in df.columns:
        st.warning("⚠️ No valid metrics found")
        return

    # Empty Values set to zero - futur work - ongoing
    metrics = ["precision_10", "latency", "drift_score"]
    for col in metrics:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0.0)

    # Radar Chart
    df_latest = df.sort_values("start_time", ascending=False).head(5)
    df_norm = df_latest.copy()
    for col in metrics:
        max_val = df_latest[col].max()
        df_norm[col] = df_latest[col] / max_val if max_val > 0 else 0

    fig_radar = go.Figure()
    for _, row in df_norm.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row[metrics].values,
            theta=metrics,
            fill='toself',
            name=row["version"]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="📉 Model Comparison: Precision, Latency, Drift (normalized)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar Chart – Top 5 for Precision
    df_bar = df.sort_values("precision_10", ascending=False).head(5)
    fig_bar = px.bar(
        df_bar,
        x="version",
        y="precision_10",
        color="drift_score",
        title="📊 Top 5 Models by Precision@10",
        labels={"precision_10": "Precision@10", "drift_score": "Drift"},
        barmode="group"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def query_prometheus_metric(metric_name):
    try:
        prometheus_url = os.getenv("PROM_URL", "http://localhost:9090")  # or http://prometheus:9090 in Docker
        response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": metric_name})
        if response.status_code == 200:
            result = response.json()["data"]["result"]
            return result
        else:
            st.warning(f"Prometheus Error: {response.status_code}")
    except Exception as e:
        st.error(f"Prometheus-Query failed: {e}")
    return []


def safe_get(metric_result):
    try:
        return float(metric_result[0]["value"][1])
    except (IndexError, KeyError, TypeError):
        return None

def show_prometheus_metrics():
    
        prometheus_url = os.getenv("PROM_URL", "http://localhost:9090")

        def q(metric):
            return query_prometheus_metric(metric)

        latency_best = safe_get(q('recommendation_latency_seconds_sum{model="Deep Hybrid-KNN_best"}'))
        latency_local = safe_get(q('recommendation_latency_seconds_sum{model="Deep Hybrid-KNN_local"}'))

        count_best = safe_get(q('recommendation_requests_total{model="Deep Hybrid-KNN_best"}'))
        count_local = safe_get(q('recommendation_requests_total{model="Deep Hybrid-KNN_local"}'))

        error_405 = safe_get(q('error_count_total{endpoint="/recommend",status_code="405"}'))
        error_500 = safe_get(q('error_count_total{endpoint="/recommend",status_code="500"}'))

        total_requests = safe_get(q('request_count_total{endpoint="/recommend"}'))
        unique_users = safe_get(q("recommendation_unique_users_total"))
        precision = safe_get(q('model_precision_at_10{model="Deep Hybrid-KNN_best"}'))
        health = safe_get(q("health_status"))

        col1, col2 = st.columns(2)

        with col1:
            if latency_best is not None:
                st.metric("🕒 Latency (Best)", f"{latency_best:.2f}s")
            if count_best is not None:
                st.metric("📬 Requests (Best)", int(count_best))
            if error_405 is not None:
                st.metric("❗ Errors  (405)", int(error_405))
            if precision is not None:
                st.metric("🎯 Precision@10", f"{precision:.4f}")

        with col2:
            if latency_local is not None:
                st.metric("🕒 Latency (Local)", f"{latency_local:.2f}s")
            if count_local is not None:
                st.metric("📬 Requests (Local)", int(count_local))
            if error_500 is not None:
                st.metric("❗ Errors  (500)", int(error_500))
            if unique_users is not None:
                st.metric("👥 Interactions", int(unique_users))

        if health is not None:
            st.metric("❤️ Health Status", "✅ OK" if int(health) == 1 else "❌ Down")

    # Separate Drift-Block

def generate_drift_report():
    try:
        result = subprocess.run(
            ["python", "src/monitoring/generate_drift_report_extended.py"],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("📄 Drift report successfully generated.")
        st.code(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("❌ Error generating the drift report.")
        st.code(e.stderr)


#------------------------------------------------------------------------------------------------------    

def show_admin_panel():
    st.header("👑 Admin Panel & Pipeline Options")

    # Pipeline-Parameter
    test_user_count = st.slider("Amount Test-User for Validation", 10, 200, 100)
   # main parameter 
    n_neighbors = st.slider("KNN neighbors (n_neighbors)", 5, 100, 12)
    latent_dim = st.slider("Latent dimension (latent_dim)", 8, 128, 32, step=8)
    hidden_dim = st.slider("Hidden layer size (hidden_dim)", 16, 512, 256, step=16)
    tfidf_features = st.slider("TF-IDF max_features (tfidf_features)", 50, 3000, 300)
    epochs = st.slider("Epochs", 5, 100, 30, step=5)


    col1, col2, col3 = st.columns(3)

    with col1:
        batch_size = st.selectbox("📦 Batch Size", [16, 32, 64, 128, 256], index=3)

    with col2:
        metric = st.selectbox("📏 Distance Metric", ["cosine", "euclidean", "manhattan"], index=0)

    with col3:
        lr = st.select_slider("🚀 Learning Rate (lr)", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    
    col1, col2 = st.columns(2)
    with col1:
        content_weight = st.slider("Content Weight", 0.0, 1.0, 0.5, step=0.05)
        collab_weight = round(1.0 - content_weight, 2)
        st.markdown(f"**Collaborative Weight:** `{collab_weight}` (auto-adjusted)")

    with col2:
        power_factor = st.slider("Power Factor", 0.1, 5.0, 1.0, step=0.1)
        drop_threshold = st.slider("Drop Threshold", 0.0, 1.0, 0.0, step=0.05)
    # Save
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
    with st.expander("🔍 Explain Parameters"):
        st.markdown("""
    | **Parameter**           | **Detailed Description** |
    |-------------------------|---------------------------|
    | **`test_user_count`**   | Number of users used during validation. A higher number gives more robust evaluation but increases computation time. |
    | **`n_neighbors`**       | Determines how many nearby items or users are considered when making recommendations using K-Nearest Neighbors (KNN). A higher number increases the chance of generalizing well to different users but may dilute recommendation quality for individual preferences. Too low can lead to overly specific or sparse recommendations. |
    | **`latent_dim`**        | Defines the number of dimensions in the compressed latent space learned by the autoencoder. This space captures user or item features in a reduced format. More dimensions mean richer representations, but also a higher risk of overfitting and slower training. Typical values range from 16 to 128. |
    | **`hidden_dim`**        | Sets the size of the hidden layer in the neural network, which lies between the input and the latent layer. Larger hidden layers can model more complex patterns but increase computational cost and overfitting risk. |
    | **`tfidf_features`**    | Specifies how many of the top TF-IDF ranked terms (e.g., tags, genres, keywords) to include in the content-based vector representation. Increasing this provides more detailed content embeddings but also leads to higher dimensionality and sparsity. |
    | **`epochs`**            | The number of times the model sees the entire training dataset during learning. More epochs allow for better learning, but after a certain point, the model may overfit and memorize patterns rather than generalize. Use early stopping or validation to prevent this. |
    | **`batch_size`**        | Controls how many samples are processed together before the model updates its internal parameters. Larger batches stabilize learning but require more memory. Smaller batches introduce noise into training, which can sometimes help generalization. |
    | **`lr (learning rate)`**| A critical hyperparameter that determines how big a step the optimizer takes when updating weights. A small value like `0.0001` ensures slow and stable learning. Larger values accelerate training but can overshoot optimal solutions and cause instability. |
    | **`metric`**            | Defines how similarity between vectors is calculated. Common options include `cosine` (angle-based, scale-invariant), `euclidean` (straight-line distance), etc. The chosen metric impacts how similar users or items appear in the model space. |
    | **`content_weight`**    | Controls how much the content-based similarity (from genres, tags, descriptions) contributes to the final recommendation score. A value of `1.0` means pure content-based filtering, while `0.0` means content is ignored. Typically combined with `collab_weight` to balance hybrid models. |
    | **`collab_weight`**     | Determines the influence of collaborative filtering (based on user behavior like ratings or interactions) in the hybrid recommendation. Must complement `content_weight`, and the two often sum to 1.0 for weighted blending. |
    | **`power_factor`**      | Applies an exponent to the final recommendation scores before ranking. Values greater than 1.0 amplify strong similarities and sharpen rankings; values less than 1.0 smooth the score distribution. This can be useful for controlling diversity. |
    | **`drop_threshold`**    | Acts as a cutoff filter. If the final recommendation score is below this threshold, the item will be excluded from results. Useful to suppress uncertain or weak recommendations. Example: `0.2` means "only show items with at least 20% confidence." |
        """)
            
    data_dir = os.getenv("DATA_DIR", "/app/data")
    config_path = os.path.join(data_dir, "processed", "pipeline_conf.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    st.success("✅ Pipeline configuration saved!")
 # 🔁 DAG Trigger-UI 
    for dag_id in DAGS:
        label = DAGS[dag_id]["label"]
        dag_toggle_ui(dag_id)
        if get_dag_status(dag_id):
            btn_label = f"▶️ Start {label} (DAG: {dag_id})"
            if st.button(btn_label, key=f"run_{dag_id}"):
             
                requests.post(f"{API_URL}/airflow/abort-runs?dag_id={dag_id}")
                abort_resp = requests.post(f"{API_URL}/airflow/abort-runs?dag_id={dag_id}")
                if abort_resp.ok:
                    st.info("⏹️ Old DAG runs have been stopped.")
                else:
                    st.warning(f"⚠️ Aborting was not successful: {abort_resp.text}")
                resp = trigger_dag(dag_id, st.session_state["pipeline_conf"])
                if resp.status_code in (200, 201):
                    st.success(f"{label} gestartet!")
                    st.session_state[f"{dag_id}_triggered"] = True
                else:
                    st.error(resp.text)
        else:
            st.info(f"ℹ️ {label} is **disabled** – activate it above to run.")

        if st.session_state.get(f"{dag_id}_triggered", False):
            show_dag_progress(dag_id)
            st.session_state[f"{dag_id}_triggered"] = False

    # 🔧 BentoML Service UI 
    with st.expander("🛠️ BentoML Service (Docker)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ BentoML Service START", key="bento_start"):
                os.system("docker compose up -d bentoml_service")
                st.success("BentoML-Service started")
                st.experimental_rerun()
        with col2:
            if st.button("🛑 BentoML Service STOP", key="bento_stop"):
                os.system("docker compose stop bentoml_service")
                st.warning("BentoML-Service stoppted")
                st.experimental_rerun()

        def is_bento_running():
            import subprocess
            result = subprocess.run(
                "docker ps --filter 'name=bentoml_service' --filter 'status=running' --format '{{.Names}}'",
                shell=True, capture_output=True, text=True
            )
            return "bentoml_service" in result.stdout

        if is_bento_running():
            st.success("🟢 BentoML service is running")
        else:
            st.error("🔴 BentoML service is stopped (manually or automatically).")

    # 📊 MLflow Registry Metrics
    st.markdown("""
    <div style="
        background-color:#e3f2fd;
        color:#0d47a1;
        padding:10px 16px;
        font-size:20px;
        font-weight:bold;
        border-radius:10px 10px 0 0;
        border: 1px solid #90caf9;
        border-bottom:none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    ">
    (DagsHub MLFLOW)
    </div>
    """, unsafe_allow_html=True)

    # 👇 the real expander
    with st.expander("📊 Show Registry-Models & Tags", expanded=False):
        try:
            show_registry_metrics()
        except Exception as e:
            st.error(f"Error loading registry metrics: {e}")
    
    with st.expander("🧪 Monitoring Results", expanded=False):
        show_drift_score()

    with st.expander("📡 Grafana Dashboard", expanded=False):
        show_grafana_dashboard()

    with st.expander("📁 Monitoring Downloads", expanded=False):
        show_monitoring_downloads()

    # === Streamlit UI ===
    with st.expander("📊 Model Drift Monitoring - Snapshot Drift. Train & Test", expanded=False):
        if os.path.exists(drift_html_path):
            with open(drift_html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=1000, scrolling=True)
        else:
            st.warning("⚠️ No Evidently Drift Report found.")

    with st.expander("📊 Model Drift Monitoring - Extended Current VS Best Model", expanded=False):
        if os.path.exists(drift_html_pathext):
            with open(drift_html_pathext, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=1000, scrolling=True)
        else:
            st.warning("⚠️ No Evidently Drift Report found.")
    
    with st.expander("📊 MLFLOW Drift Latency", expanded=False):
        # === Lade Metriken ===
        if os.path.exists(METRICS_PATH):
            metrics = pd.read_csv(METRICS_PATH)
            st.subheader("📈 Model metrics from MLflow")
            st.dataframe(metrics.tail(10))

            # === Plot: Precision@10 ===
            if "precision_10" in metrics.columns:
                fig = px.line(metrics, x="start_time", y="precision_10", title="Precision@10 Verlauf")
                st.plotly_chart(fig)

            # === Plot: Drift Score ===
            if "drift_score" in metrics.columns:
                fig = px.line(metrics, x="start_time", y="drift_score", title="Drift Score Verlauf")
                st.plotly_chart(fig)

            # === Plot: Latency ===
            if "latency" in metrics.columns:
                fig = px.line(metrics, x="start_time", y="latency", title="Inference Latency")
                st.plotly_chart(fig)
        else:
            st.warning("⚠️ No metrics found under metrics_from_mlflow.csv")           
    

    with st.expander("🔄 Refresh Precision & Metrics from MLflow", expanded=False):
        try:    
            df = load_precision_history()
            plot_and_save_precision(df)
            save_metrics_csv()
            show_model_comparison_charts()  
            st.success("📊 Metrics were refreshed, saved, and visualized from MLflow.")
        except Exception as e:
            st.error(f"Error Precision metrics: {e}")
    

    with st.expander("📈 Drift Monitoring", expanded=False):
        try:
            response = requests.get("http://api_service:8000/metrics/drift", timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                metrics = {}
                for line in lines:
                    if line.startswith("#") or line.strip() == "":
                        continue
                    key, value = line.split()
                    metrics[key] = float(value)

                st.metric("🔄 Data Drift Share", f"{metrics.get('data_drift_share', 0):.2%}")
                st.metric("🎯 Target Drift PSI", f"{metrics.get('target_drift_psi', 0):.4f}")
                st.metric("📊 Drifted Columns", int(metrics.get("drifted_columns_total", 0)))
                st.metric("⚠️ Drift Alert", "🚨 YES" if metrics.get("drift_alert", 0) == 1 else "✅ NO")

            else:
                st.warning("Drift metrics could not be loaded.")
        except Exception as e:
            st.error(f"Error retrieving drift metrics: {e}")
    
    with st.expander("📊 Prometheus Monitoring", expanded=False):
        show_prometheus_metrics()