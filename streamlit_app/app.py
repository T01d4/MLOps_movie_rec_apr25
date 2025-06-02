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

def get_tmdb_poster_url(movie_id, links_df, api_key):
    try:
        tmdb_row = links_df[links_df["movieId"] == movie_id]
        if not tmdb_row.empty and not pd.isna(tmdb_row["tmdbId"].values[0]):
            tmdb_id = int(tmdb_row["tmdbId"].values[0])
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
            resp = requests.get(url)
            if resp.status_code == 200:
                poster_path = resp.json().get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w342{poster_path}"
        return None
    except Exception as e:
        return None
    

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
    data_available = True
except Exception as e:
    st.warning(f"❌ Filme konnten nicht geladen werden: {e}")
    data_available = False

if data_available:
    selected_movies = [
        st.selectbox("Film 1", [""] + movie_titles, key="film1"),
        st.selectbox("Film 2", [""] + movie_titles, key="film2"),
        st.selectbox("Film 3", [""] + movie_titles, key="film3"),
        st.selectbox("Film 4 (optional)", [""] + movie_titles, key="film4"),
        st.selectbox("Film 5 (optional)", [""] + movie_titles, key="film5")
    ]
    selected_movies = [f for f in selected_movies if f]


    links_df = pd.read_csv("data/raw/links.csv")
    api_key = os.getenv("TMDB_API_KEY")

    def filter_by_genre(recommended_ids, selected_movies, movies_df):
        selected_genres = set()
        for title in selected_movies:
            genres = movies_df[movies_df["title"] == title]["genres"]
            if not genres.empty:
                selected_genres.update([g.strip() for g in genres.values[0].split('|')])
        result = []
        for mid in recommended_ids:
            movie_genres = movies_df[movies_df["movieId"] == mid]["genres"]
            if not movie_genres.empty:
                if any(g in movie_genres.values[0].split('|') for g in selected_genres):
                    result.append(mid)
        return result

    if len(selected_movies) >= 3:
        st.success("✅ Auswahl ausreichend – Empfehlung kann gestartet werden.")
        if st.button("Empfehle 10 ähnliche Filme (User-Based Neighbors, optimiert)"):
            try:
                # Daten laden
                matrix_path = "data/processed/user_matrix.csv"
                movies_path = "data/raw/movies.csv"
                ratings_path = "data/raw/ratings.csv"
                model_path = "models/user_model.pkl"
                links_path = "data/raw/links.csv"
                api_key = os.getenv("TMDB_API_KEY")

                user_df = pd.read_csv(matrix_path, index_col=0)
                movies_df = pd.read_csv(movies_path)
                ratings_df = pd.read_csv(ratings_path)
                links_df = pd.read_csv(links_path) if os.path.exists(links_path) else None

                # 1. Filme der Auswahl (MovieIDs)
                selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
                if not selected_movie_ids:
                    st.warning("Keine gültigen Filme in User-Modell gefunden.")
                    st.stop()

                # 2. User, die mindestens X deiner Filme bewertet haben
                min_overlap = 2  # Du kannst hier 3 oder mehr testen!
                user_movie_counts = ratings_df[ratings_df["movieId"].isin(selected_movie_ids)].groupby("userId")["movieId"].nunique()
                matching_users = user_movie_counts[user_movie_counts >= min_overlap].index.tolist()
                if not matching_users:
                    st.warning("Keine passenden User mit ausreichend Überlappung gefunden. Verringere min_overlap oder wähle andere Filme.")
                    st.stop()

                # 3. Query-User-Vektor als Mittelwert dieser User-Vektoren
                user_vec = user_df.loc[user_df.index.intersection(matching_users)].values.mean(axis=0).reshape(1, -1)

                # 4. KNN: ähnlichste User (neighbors) finden
                with open(model_path, "rb") as f:
                    knn_user = pickle.load(f)
                _, user_neighbor_indices = knn_user.kneighbors(user_vec, n_neighbors=15)
                similar_user_ids = [user_df.index[idx] for idx in user_neighbor_indices[0]]

                # 5. Empfehlungen: Nur Filme, die diese User hoch bewertet haben, aber du nicht ausgewählt hast
                neighbors_ratings = ratings_df[
                    (ratings_df["userId"].isin(similar_user_ids)) & 
                    (~ratings_df["movieId"].isin(selected_movie_ids))
                ]
                # Filme, die von möglichst vielen der ähnlichen Nutzer bewertet wurden
                movie_counts = neighbors_ratings.groupby("movieId")["rating"].agg(["mean", "count"])
                # Score: erst nach Anzahl der Bewertungen, dann Durchschnittsbewertung sortieren
                movie_counts = movie_counts.sort_values(by=["count", "mean"], ascending=[False, False])
                top_movie_candidates = movie_counts.head(30).index.tolist()

                # Genrefilter wie gehabt
                selected_genres = set()
                for mid in selected_movie_ids:
                    genres = movies_df[movies_df["movieId"] == mid]["genres"]
                    if not genres.empty:
                        selected_genres.update(genres.iloc[0].split("|"))

                recommended = movies_df[movies_df["movieId"].isin(top_movie_candidates)].copy()
                if not recommended.empty and selected_genres:
                    recommended = recommended[recommended["genres"].apply(
                        lambda g: any(genre in g.split("|") for genre in selected_genres)
                    )]

                st.subheader("🎬 Top 10 Empfehlungen (User-Based Neighbors, optimiert)")
                if not recommended.empty:
                    for _, row in recommended.head(10).iterrows():
                        poster_url = None
                        if links_df is not None:
                            poster_url = get_tmdb_poster_url(row["movieId"], links_df, api_key)
                        cols = st.columns([1, 5])
                        with cols[0]:
                            if poster_url:
                                st.image(poster_url, width=100)
                            else:
                                st.write("Kein Poster")
                        with cols[1]:
                            st.markdown(f"**{row['title']}**")
                            st.write(f"Genres: {row['genres']}")
                else:
                    st.info("⚠️ Keine passenden Empfehlungen gefunden. Probiere andere Auswahl.")

            except Exception as e:
                st.error(f"❌ Fehler bei user-based Empfehlung: {e}")
 

        #hybrid-modell
        if st.button("Empfehle 10 ähnliche Filme (MLflow-Bestmodell, Genre-dynamisch)"):
            try:
                # Laden...
                matrix_path = "data/processed/hybrid_matrix.csv"
                movies_path = "data/raw/movies.csv"
                model_path = "models/hybrid_model.pkl"

                # MovieId als Index!
                hybrid_df = pd.read_csv(matrix_path, index_col=0)
                movies = pd.read_csv(movies_path)

                # Finde MovieIds deiner Auswahl
                selected_movie_ids = movies[movies["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()

                # Zeige Genres der Auswahl
                selected_genres = set()
                for mid in selected_movie_ids:
                    g = movies[movies["movieId"] == mid]["genres"].iloc[0]
                    selected_genres.update(g.split("|"))
                st.write(f"⚡️ Genres deiner Auswahl: {selected_genres}")

                # Finde Indexe im hybrid_df
                selected_indices = [hybrid_df.index.get_loc(mid) for mid in selected_movie_ids if mid in hybrid_df.index]
                if not selected_indices:
                    st.warning("Keine gültigen Filme gefunden.")
                    st.stop()

                # Lade Modell
                with open(model_path, "rb") as f:
                    knn = pickle.load(f)

                # Nutzervektor
                user_vec = hybrid_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
                _, rec_indices = knn.kneighbors(user_vec, n_neighbors=10 + len(selected_indices))
                rec_movie_ids = [hybrid_df.index[idx] for idx in rec_indices[0] if hybrid_df.index[idx] not in selected_movie_ids]

                # Filtere Empfehlungen nach dynamischen Genres
                recommended_movies = movies[movies["movieId"].isin(rec_movie_ids)]
                filtered = recommended_movies[recommended_movies["genres"].apply(
                    lambda g: any(genre in g.split("|") for genre in selected_genres)
                )]
                st.subheader("🎬 Top 10 Empfehlungen (MLflow-Bestmodell, Genre-dynamisch)")
                if not filtered.empty:
                    for _, row in filtered.head(10).iterrows():
                        poster_url = get_tmdb_poster_url(row["movieId"], links_df, api_key)
                        cols = st.columns([1, 5])
                        with cols[0]:
                            if poster_url:
                                st.image(poster_url, width=100)
                            else:
                                st.write("Kein Poster")
                        with cols[1]:
                            st.markdown(f"**{row['title']}**")
                            st.write(f"Genres: {row['genres']}")
                else:
                    st.info("⚠️ Keine passenden Empfehlungen gefunden. Probiere andere Filme.")

            except Exception as e:
                st.error(f"❌ Fehler bei dynamischer Genre-Empfehlung: {e}")   
        #basic model            
        if st.button("Empfehle 10 ähnliche Filme (lokal, wie Hybrid-Modell)"):
            try:
                # --- Pfade wie oben ---
                movies_path = "data/raw/movies.csv"
                tags_path = "data/raw/tags.csv"
                scores_path = "data/raw/genome-scores.csv"

                # --- Lade Daten ---
                movies = pd.read_csv(movies_path)
                tags = pd.read_csv(tags_path).dropna(subset=["tag"])
                scores = pd.read_csv(scores_path)

                # --- Kombiniere Genres & Tags wie im Preprocessing ---
                tags_combined = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(t)).reset_index()
                movies = pd.merge(movies, tags_combined, on="movieId", how="left")
                movies["combined"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"].fillna("")

                tfidf = TfidfVectorizer(max_features=300, stop_words="english")
                content_embeddings = tfidf.fit_transform(movies["combined"])

                collab = scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

                # Nur Movies mit genome-scores verwenden (wie hybrid_matrix)
                valid_movies = movies[movies["movieId"].isin(collab.index)].copy()
                content_embeddings = content_embeddings[[i for i, mid in enumerate(movies["movieId"]) if mid in valid_movies["movieId"].values]]
                collab = collab.loc[valid_movies["movieId"]]

                # Skaliere beide Matrizen
                scaler = MinMaxScaler()
                collab_scaled = scaler.fit_transform(collab)
                content_scaled = scaler.fit_transform(content_embeddings.toarray())
                hybrid_matrix = np.hstack([collab_scaled, content_scaled])

                # === Index == MovieId! ===
                hybrid_df = pd.DataFrame(hybrid_matrix, index=valid_movies["movieId"])

                # --- User-Auswahl umsetzen wie oben ---
                selected_movie_ids = valid_movies[valid_movies["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()

                selected_genres = set()
                for mid in selected_movie_ids:
                    g = valid_movies[valid_movies["movieId"] == mid]["genres"].iloc[0]
                    selected_genres.update(g.split("|"))
                st.write(f"⚡️ Genres deiner Auswahl: {selected_genres}")

                # Index in der Matrix suchen
                selected_indices = [hybrid_df.index.get_loc(mid) for mid in selected_movie_ids if mid in hybrid_df.index]
                if not selected_indices:
                    st.warning("Keine gültigen Filme gefunden.")
                    st.stop()

                # --- Modell auf dem neuen Schema trainieren ---
                knn = NearestNeighbors(n_neighbors=10 + len(selected_indices), metric="cosine")
                knn.fit(hybrid_df.values)

                # --- User-Vektor analog ---
                user_vec = hybrid_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
                _, rec_indices = knn.kneighbors(user_vec)
                rec_movie_ids = [hybrid_df.index[idx] for idx in rec_indices[0] if hybrid_df.index[idx] not in selected_movie_ids]

                # --- Empfehlungen wie oben filtern ---
                recommended_movies = valid_movies[valid_movies["movieId"].isin(rec_movie_ids)]
                filtered = recommended_movies[recommended_movies["genres"].apply(
                    lambda g: any(genre in g.split("|") for genre in selected_genres)
                )]

                st.subheader("🎬 Top 10 Empfehlungen (Lokal, Hybrid-Style)")
                if not filtered.empty:
                    for _, row in filtered.head(10).iterrows():
                        poster_url = get_tmdb_poster_url(row["movieId"], links_df, api_key)
                        cols = st.columns([1, 5])
                        with cols[0]:
                            if poster_url:
                                st.image(poster_url, width=100)
                            else:
                                st.write("Kein Poster")
                        with cols[1]:
                            st.markdown(f"**{row['title']}**")
                            st.write(f"Genres: {row['genres']}")
                else:
                    st.info("⚠️ Keine passenden Empfehlungen gefunden. Probiere andere Filme.")

            except Exception as e:
                st.error(f"❌ Fehler bei der lokalen Hybrid-Empfehlung: {e}")
else:
    st.info("ℹ️ Noch keine Filmdaten verfügbar. Bitte generiere die Daten per Admin-Panel.")
    
# === ADMIN-Bereich ===
if role == "admin":


    # --- Erweiterte Admin-Optionen ---
    st.subheader("⚙️ Pipeline-Optionen")

    # (A) Erzwinge neue Datenverarbeitung
    #force_rebuild = st.checkbox("Neues Dataset (Force rebuild)?", value=False)

    # (B) Test-User-Konfiguration für Validierung
    test_user_count = st.slider("Anzahl Test-User für Validierung", 10, 200, 100)


    # (D) Parameter-Tuning
    n_neighbors = st.slider("KNN Nachbarn (n_neighbors)", 1, 50, 10)
    tfidf_features = st.slider("TF-IDF max_features", 50, 2000, 300)

    # Optionen speichern (z. B. als dict in st.session_state, um sie in der Trigger-API zu übergeben)
    st.session_state["pipeline_conf"] = {
        #"force_rebuild": force_rebuild,
        "test_user_count": test_user_count,
        "n_neighbors": n_neighbors,
        "tfidf_features": tfidf_features,
    }
    st.subheader("⚙️ Admin Panel – Airflow & MLflow")

    col1, col2 = st.columns(2)

    if "dag_triggered" not in st.session_state:
        st.session_state["dag_triggered"] = False

    with col1:
        #update Pipeline
        if st.button("▶️ Starte DAG: movie_recommendation_pipeline"):
            try:
                # 1. Laufende DAG-Runs finden und stoppen!
                dag_id = "movie_recommendation_pipeline"
                airflow_url = API_URL
                auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

                # Step 1: Suche alle laufenden Runs
                running_runs_url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns?state=running"
                runs_resp = requests.get(running_runs_url, auth=auth)
                runs_resp.raise_for_status()
                running_runs = runs_resp.json().get("dag_runs", [])

                # Step 2: Setze alle laufenden Runs auf "failed"
                for run in running_runs:
                    run_id = run["dag_run_id"]
                    patch_url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
                    patch_resp = requests.patch(patch_url, json={"state": "failed"}, auth=auth)
                    if patch_resp.status_code == 200:
                        st.info(f"❗ Abgebrochen: Run {run_id}")
                    else:
                        st.warning(f"⚠️ Konnte Run {run_id} nicht abbrechen: {patch_resp.text}")

                # Step 3: Starte neuen DAG-Run
                response = requests.post(
                    f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns",
                    json={"conf": st.session_state["pipeline_conf"]},
                    auth=auth
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
    #Charts Mlflow
        #
        if st.button("📊 Zeige MLflow Metriken"):
            try:
                import plotly.graph_objects as go
                import plotly.express as px

                exp_name = "model_validation"
                df = mlflow.search_runs(experiment_names=[exp_name])

                # --- Preprocessing ---
                df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
                df = df[df["start_time"].notna()].copy()

                # Metriken zu Prozent (falls gewünscht)
                for col in ["metrics.precision_10_hybrid", "metrics.precision_10_user"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        df[col] = df[col] * 100  # Prozent!

                df = df.dropna(subset=["metrics.precision_10_hybrid", "metrics.precision_10_user"], how="all")
                df = df.sort_values("start_time", ascending=True)

                # Modell-Version extrahieren (aus Run-Name oder Model Registry Tag)
                if "tags.mlflow.runName" in df.columns:
                    df["model_version"] = (
                        df["tags.mlflow.runName"].str.extract(r'(\d+)$').astype(float)
                    )
                elif "tags.mlflow.modelVersion" in df.columns:
                    df["model_version"] = pd.to_numeric(df["tags.mlflow.modelVersion"], errors="coerce")
                else:
                    df["model_version"] = np.arange(1, len(df) + 1)

                relevant_cols = ["metrics.precision_10_hybrid", "metrics.precision_10_user"]

                # --- Tabelle ---
                st.dataframe(df[["start_time", "model_version"] + relevant_cols].sort_values("start_time", ascending=False))

                # --- Chart 1: Zeitverlauf als Liniendiagramm ---
                fig = go.Figure()
                if "metrics.precision_10_hybrid" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["start_time"],
                        y=df["metrics.precision_10_hybrid"],
                        mode="lines+markers",
                        name="Precision@10 Hybrid",
                        line=dict(color="#3778C2", width=2),
                        marker=dict(size=7, symbol="diamond")
                    ))

                fig.update_layout(
                    title="⏳ Precision@10: Zeitverlauf",
                    xaxis_title="Run Start Time",
                    yaxis_title="Precision@10 (%)",
                    legend=dict(orientation="h", x=0, y=1.12),
                    height=350,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

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
            "train_user_model", "train_hybrid_model",
            "validate_models", "predict_best_model"
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
            st.rerun()