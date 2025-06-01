# streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os
import requests
from dotenv import load_dotenv
from auth import authenticate_user
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from requests.auth import HTTPBasicAuth

load_dotenv(".env")

required_env_vars = [
    "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD", "API_URL"
]

missing = [v for v in required_env_vars if os.getenv(v) is None]
if missing:
    st.error(f"❌ Fehlende .env Einträge: {missing}")
    st.stop()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
API_URL = os.getenv("API_URL")
AIRFLOW_USER = "admin"
AIRFLOW_PASS = "admin"

def fetch_dag_task_statuses():
    try:
        airflow_url = os.getenv("API_URL", "http://localhost:8080")
        dag_id = "movie_recommendation_pipeline"
        auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

        run_url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"
        run_resp = requests.get(run_url, auth=auth)
        run_resp.raise_for_status()
        dag_run_id = run_resp.json()["dag_runs"][0]["dag_run_id"]

        task_url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        task_resp = requests.get(task_url, auth=auth)
        task_resp.raise_for_status()
        return task_resp.json()["task_instances"], dag_run_id

    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der DAG-Task-Status: {e}")
        return [], None

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

with st.sidebar:
    st.title("🔐 Login")
    username = st.text_input("Benutzername")
    password = st.text_input("Passwort", type="password")
    if st.button("Login"):
        role = authenticate_user(username, password)
        st.session_state["role"] = role

    try:
        health = requests.get(f"{API_URL}/health", auth=HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS), timeout=3)
        if health.status_code == 200:
            st.success("Airflow API: ✅ erreichbar")
        else:
            st.warning("Airflow API: ⚠️ antwortet nicht richtig")
    except:
        st.error("Airflow API: ❌ nicht erreichbar")

role = st.session_state.get("role", None)
if not role:
    st.warning("Bitte zuerst einloggen")
    st.stop()

st.success(f"Angemeldet als: {role.upper()}")

try:
    movies_df = pd.read_csv("data/raw/movies.csv")
    movie_titles = sorted(movies_df["title"].dropna().unique())
except Exception as e:
    st.error(f"❌ Fehler beim Laden der Filme: {e}")
    st.stop()

selected_movies = [
    st.selectbox("Film 1", [""] + movie_titles, key="film1"),
    st.selectbox("Film 2", [""] + movie_titles, key="film2"),
    st.selectbox("Film 3", [""] + movie_titles, key="film3"),
    st.selectbox("Film 4 (optional)", [""] + movie_titles, key="film4"),
    st.selectbox("Film 5 (optional)", [""] + movie_titles, key="film5")
]
selected_movies = [f for f in selected_movies if f]

if len(selected_movies) >= 3:
    st.success("✅ Auswahl ausreichend – Empfehlung kann gestartet werden.")
    if st.button("Empfehle 10 ähnliche Filme (lokal)"):
        try:
            movies_df = pd.read_csv("data/raw/movies.csv").dropna()
            tags_df = pd.read_csv("data/raw/tags.csv").dropna(subset=["tag"])
            scores_df = pd.read_csv("data/raw/genome-scores.csv")

            tags_combined = tags_df.groupby("movieId")["tag"].apply(lambda t: " ".join(t)).reset_index()
            movies_df = pd.merge(movies_df, tags_combined, on="movieId", how="left")
            movies_df["combined"] = movies_df["genres"].str.replace("|", " ", regex=False) + " " + movies_df["tag"].fillna("")

            tfidf = TfidfVectorizer(max_features=300, stop_words="english")
            content_embeddings = tfidf.fit_transform(movies_df["combined"])

            collab = scores_df.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

            common_ids = movies_df[movies_df["movieId"].isin(collab.index)].copy()
            content_embeddings = content_embeddings[[i for i, mid in enumerate(movies_df["movieId"]) if mid in common_ids["movieId"].values]]
            collab = collab.loc[common_ids["movieId"]]

            scaler = MinMaxScaler()
            collab_scaled = scaler.fit_transform(collab)
            content_scaled = scaler.fit_transform(content_embeddings.toarray())
            hybrid_matrix = np.hstack([collab_scaled, content_scaled])

            knn = NearestNeighbors(n_neighbors=11, metric="cosine")
            knn.fit(hybrid_matrix)

            selected_indices = []
            for title in selected_movies:
                match = common_ids[common_ids["title"].str.lower() == title.lower()]
                if not match.empty:
                    selected_indices.append(match.index[0])

            if not selected_indices:
                st.warning("⚠️ Keine gültigen Filme gefunden für Empfehlung.")
                st.stop()

            user_vector = hybrid_matrix[selected_indices].mean(axis=0).reshape(1, -1)
            distances, indices = knn.kneighbors(user_vector, n_neighbors=20)
            recommended_ids = common_ids.iloc[indices[0][1:]]["movieId"].tolist()

            all_titles = movies_df[movies_df["movieId"].isin(recommended_ids)]["title"].tolist()
            recommended_titles = [t for t in all_titles if t not in selected_movies][:10]

            st.subheader("🎬 Top 10 Empfehlungen")
            st.write(pd.DataFrame(recommended_titles, columns=["Empfohlene Titel"]))

        except Exception as e:
            st.error(f"❌ Fehler bei der Empfehlung (lokal): {e}")

# === ADMIN-Bereich ===
if role == "admin":
    st.subheader("⚙️ Admin Panel – Airflow & MLflow")

    col1, col2 = st.columns(2)

    if "dag_triggered" not in st.session_state:
        st.session_state["dag_triggered"] = False

    with col1:
        if st.button("▶️ Starte DAG: movie_recommendation_pipeline"):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/dags/movie_recommendation_pipeline/dagRuns",
                    json={"conf": {}},
                    auth=HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)
                )
                if response.status_code == 200:
                    st.success("✅ DAG wurde erfolgreich getriggert")
                    st.session_state["dag_triggered"] = True
                    st.session_state["progress"] = 0
                else:
                    st.error(f"❌ Fehler beim Triggern des DAGs: {response.text}")
            except Exception as e:
                st.error(f"❌ Fehler bei API-Aufruf: {e}")

    with col2:
        if st.button("📊 Zeige MLflow Metriken"):
            try:
                exp_name = "model_validation"
                df = mlflow.search_runs(experiment_names=[exp_name])

                # Zeitstempel zu datetime casten, Fehler zu NaT
                df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
                df = df[df["start_time"].notna()].copy()

                # Nur numerische Metriken zulassen, alle Fehler zu np.nan
                for col in ["metrics.precision_10_hybrid", "metrics.precision_10_user"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")  # Fehler zu NaN

                # OPTIONAL: Zeilen mit komplett fehlenden Metriken rauswerfen
                df = df.dropna(subset=["metrics.precision_10_hybrid", "metrics.precision_10_user"], how="all")

                # Jetzt sollte sort_values sauber laufen
                df = df.sort_values("start_time", ascending=False)

                available_metrics = [col for col in df.columns if "metrics." in col]
                relevant_cols = [c for c in ["metrics.precision_10_hybrid", "metrics.precision_10_user"] if c in available_metrics]

                if not relevant_cols:
                    st.warning("⚠️ Keine Metriken gefunden in model_validation.")
                else:
                    st.dataframe(df[["start_time"] + relevant_cols])

                # ... Rest wie gehabt ...
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der MLflow-Metriken: {e}")



    # 📋 Automatisches Task-Monitoring mit Fortschrittsanzeige
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
            "train_model", "train_user_model", "train_hybrid_model",
            "debug_env", "validate_models", "predict_best_model"
        ]

        import time

        for attempt in range(40):  # 40 × 3s = 2 Minuten max.
            tasks, run_id = fetch_dag_task_statuses()
            if tasks:
                task_states = {task["task_id"]: (task["state"] or "no_status") for task in tasks}
                task_output = f"🧭 Letzter Run: `{run_id}`\n\n"

                # Fortschritt berechnen
                finished = sum(1 for t in task_order if task_states.get(t) == "success")
                total = len(task_order)
                percent = int((finished / total) * 100)
                progress_bar.progress(percent)

                for task_id in task_order:
                    status = task_states.get(task_id, "no_status")
                    emoji = status_colors.get(status, "⚪")
                    task_output += f"{emoji} `{task_id}` → **{status}**\n"

                task_placeholder.markdown(task_output)

                if finished == total:
                    st.success("🎉 Alle Tasks abgeschlossen.")
                    st.session_state["dag_triggered"] = False
                    break

            time.sleep(3)
            st.experimental_rerun()