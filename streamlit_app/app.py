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
from mlflow.tracking import MlflowClient
import requests
import logging
import subprocess
import traceback


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

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

def ensure_best_embedding_exists():
    import subprocess
    import os
    best_embedding_path = "data/processed/hybrid_deep_embedding_best.csv"
    if not os.path.exists(best_embedding_path):
        try:
            # Optional: Vorher 'git pull' machen, falls Repo im Container veraltet ist!
            subprocess.run(["dvc", "pull", best_embedding_path], check=True)
            assert os.path.exists(best_embedding_path), "DVC pull hat das File nicht erzeugt!"
        except Exception as e:
            st.error(f"DVC pull fehlgeschlagen für Best-Embedding: {e}")
            raise
    return best_embedding_path

def fetch_dag_task_statuses():
    try:
        airflow_url = os.getenv("API_URL", "http://localhost:8080")
        dag_id = "deep_models_pipeline"
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

def compare_with_tmdb(recommended_titles, api_key):
    """Vergleicht die empfohlenen Titel mit TMDb-Similar Movies."""
    def get_tmdb_movie_id(title):
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title}
        resp = requests.get(url, params=params)
        results = resp.json().get("results", [])
        return results[0]['id'] if results else None

    def get_similar_movies(movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar"
        params = {"api_key": api_key}
        resp = requests.get(url, params=params)
        return [m['title'] for m in resp.json().get("results", [])]

    overlaps = []
    for title in recommended_titles:
        tmdb_id = get_tmdb_movie_id(title)
        if tmdb_id:
            tmdb_similar = get_similar_movies(tmdb_id)
            # Schnittmenge
            overlap = set(recommended_titles) & set(tmdb_similar)
            overlaps.extend(overlap)
    return list(set(overlaps)), len(overlaps)

def get_movie_rating(title, api_key):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    resp = requests.get(url, params=params)
    results = resp.json().get("results", [])
    if results:
        movie_id = results[0]['id']
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": api_key}
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return resp.json().get("vote_average")
    return None

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
    
def show_best_model_info():
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias("hybrid_deep_model", "best_model")
        version = mv.version
        run_id = mv.run_id
        tags = mv.tags  # dict mit Tags (Strings)

        st.info(f"🔖 Aktive Modell-Version (`@best_model`): **v{version}** (Run ID: {run_id})")
        
        # Wichtige Tags direkt zeigen:
        keys_of_interest = [
            "precision_10", "n_neighbors", "latent_dim", "tfidf_features", "algorithm"
        ]
        shown = False
        for k in keys_of_interest:
            if k in tags:
                st.markdown(f"**{k}**: `{tags[k]}`")
                shown = True

        # Optional: Alle weiteren Tags (falls welche da sind)
        other_tags = {k: v for k, v in tags.items() if k not in keys_of_interest}
        if other_tags:
            with st.expander("Weitere Modell-Tags"):
                for k, v in other_tags.items():
                    st.write(f"{k}: {v}")
        elif not shown:
            st.info("ℹ️ Keine Tags an dieser Modellversion hinterlegt.")

    except Exception as e:
        st.warning(f"Kein best_model-Alias gefunden oder Fehler: {e}") 

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

    # --- Empfehlungsfunktionen ---



    def get_deep_hybrid_knn_best_recommendations(selected_movies):
        try:
            matrix_path = ensure_best_embedding_exists()
            movies_df = pd.read_csv("data/raw/movies.csv")
            embedding_df = pd.read_csv(matrix_path, index_col=0)
            selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
            selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]
            if not selected_indices:
                return []
            user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            
            # === NEU: Modell aus MLflow Registry laden ===
            deep_knn = mlflow.pyfunc.load_model("models:/hybrid_deep_model@best_model")
            print("embedding_df.columns:", list(embedding_df.columns))
            print("MLflow Signature Columns:", [input.name for input in deep_knn.metadata.get_input_schema().inputs])

            #user_df = pd.DataFrame(user_vec, columns=embedding_df.columns)
            user_df = pd.DataFrame(user_vec, columns=[f"emb_{i}" for i in range(user_vec.shape[1])]).astype(np.float32)
            user_df = pd.DataFrame(user_vec, columns=embedding_df.columns).astype(np.float32)
            expected_cols = [f"emb_{i}" for i in range(user_vec.shape[1])]
            print("Expected columns:", expected_cols)
            print("User DataFrame columns:", list(user_df.columns))
            print("user_df.columns:", list(user_df.columns))
            print("Shape user_vec:", user_vec.shape)
            print("Erste Zeile user_df:", user_df.iloc[0].values)
            rec_indices = deep_knn.predict(user_df)
            rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0] if embedding_df.index[idx] not in selected_movie_ids]
            recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            return [{"movieId": int(row["movieId"]), "title": row["title"]} for _, row in recommended.head(10).iterrows()]
        except Exception as e:
            # st.error(f"Deep-Hybrid-KNN-Fehler: {e}")
            st.error(traceback.format_exc())  # stacktrace für Details!
            print("Deep-Hybrid-KNN-Fehler:", traceback.format_exc())  # falls logs gesammelt werden
            return []


    def get_deep_hybrid_knn_local_recommendations(selected_movies):
        try:
            matrix_path = "data/processed/hybrid_deep_embedding.csv"
            model_path = "models/hybrid_deep_knn.pkl"
            movies_df = pd.read_csv("data/raw/movies.csv")
            embedding_df = pd.read_csv(matrix_path, index_col=0)
            with open(model_path, "rb") as f:
                deep_knn = pickle.load(f)
            selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
            selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]
            if not selected_indices:
                return []
            user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            _, rec_indices = deep_knn.kneighbors(user_vec, n_neighbors=10 + len(selected_indices))
            rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0] if embedding_df.index[idx] not in selected_movie_ids]
            recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            return [{"movieId": int(row["movieId"]), "title": row["title"]} for _, row in recommended.head(10).iterrows()]
        except Exception as e:
            return []

    def get_baseline_recommendations(selected_movies):
        try:
            # Baseline: klassisches Hybrid aus Genres, Tags, genome-scores, NearestNeighbors (ohne externes Modell)
            movies = pd.read_csv("data/raw/movies.csv")
            tags = pd.read_csv("data/raw/tags.csv").dropna(subset=["tag"])
            scores = pd.read_csv("data/raw/genome-scores.csv")
            tags_combined = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(t)).reset_index()
            movies = pd.merge(movies, tags_combined, on="movieId", how="left")
            movies["combined"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["tag"].fillna("")
            tfidf = TfidfVectorizer(max_features=300, stop_words="english")
            content_embeddings = tfidf.fit_transform(movies["combined"])
            collab = scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)
            valid_movies = movies[movies["movieId"].isin(collab.index)].copy()
            content_embeddings = content_embeddings[[i for i, mid in enumerate(movies["movieId"]) if mid in valid_movies["movieId"].values]]
            collab = collab.loc[valid_movies["movieId"]]
            scaler = MinMaxScaler()
            collab_scaled = scaler.fit_transform(collab)
            content_scaled = scaler.fit_transform(content_embeddings.toarray())
            hybrid_matrix = np.hstack([collab_scaled, content_scaled])
            hybrid_df = pd.DataFrame(hybrid_matrix, index=valid_movies["movieId"])
            selected_movie_ids = valid_movies[valid_movies["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
            selected_indices = [hybrid_df.index.get_loc(mid) for mid in selected_movie_ids if mid in hybrid_df.index]
            if not selected_indices:
                return []
            knn = NearestNeighbors(n_neighbors=10 + len(selected_indices), metric="cosine")
            knn.fit(hybrid_df.values)
            user_vec = hybrid_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            _, rec_indices = knn.kneighbors(user_vec)
            rec_movie_ids = [hybrid_df.index[idx] for idx in rec_indices[0] if hybrid_df.index[idx] not in selected_movie_ids]
            recommended = valid_movies[valid_movies["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            return [{"movieId": int(row["movieId"]), "title": row["title"]} for _, row in recommended.head(10).iterrows()]
        except Exception as e:
            return []
    #tmdb

    @st.cache_data(show_spinner="Lade TMDb Similar…")
    def get_tmdb_similar_recommendations_cached(selected_movie, movies_df, links_df, api_key):
        try:
            logging.info(f"Starte TMDb-Similar-Call für: {selected_movie}")
            # 1. MovieId im movies_df suchen
            row = movies_df[movies_df["title"] == selected_movie]
            if row.empty:
                row = movies_df[movies_df["title"].str.lower() == selected_movie.lower()]
                logging.info(f"Case-insensitive Suche (movies_df): {row}")
            if not row.empty:
                movie_id = row["movieId"].values[0]
            else:
                movie_id = None

            # 2. tmdbId aus links_df holen
            tmdb_id = None
            if movie_id is not None:
                row = links_df[links_df["movieId"] == movie_id]
                if not row.empty and not pd.isna(row["tmdbId"].values[0]):
                    tmdb_id = int(row["tmdbId"].values[0])
            logging.info(f"Ermittelte tmdb_id: {tmdb_id}")

            # 3. TMDb-API-Call (nur wenn ID existiert)
            if tmdb_id:
                url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/recommendations"
                params = {"api_key": api_key}
                resp = requests.get(url, params=params, timeout=8)
                logging.info(f"TMDb API-Response: {resp.status_code}")
                if resp.status_code == 200:
                    results = resp.json().get("results", [])[:10]
                    out = []
                    for movie in results:
                        poster_url = f"https://image.tmdb.org/t/p/w342{movie['poster_path']}" if movie.get("poster_path") else None
                        logging.info(f"Movie: {movie['title']} | Poster: {poster_url}")
                        out.append({
                            "title": movie["title"],
                            "tmdbId": movie["id"],
                            "poster_url": poster_url
                        })
                    return out
                else:
                    logging.error(f"TMDb API Fehler: {resp.text}")
            else:
                logging.warning("Keine tmdb_id gefunden!")
        except Exception as e:
            logging.exception("Fehler beim Laden von TMDb Similar!")
        return []

    # --- TABLE: Empfehlungen aller 6 Modelle ---
    # --- EMPFEHLUNGS-TABELLE: Nur noch Deep Hybrid, Baseline & TMDb ---
    if len(selected_movies) >= 3 and st.button("Empfehle 10 Filme"):
            show_best_model_info()
            recommendations = {
                "Deep Hybrid-KNN_best": get_deep_hybrid_knn_best_recommendations(selected_movies),
                "Deep Hybrid-KNN_local": get_deep_hybrid_knn_local_recommendations(selected_movies),
                "Basis Modell": get_baseline_recommendations(selected_movies),
                "TMDb-Similar": get_tmdb_similar_recommendations_cached(selected_movies[0], movies_df, links_df, api_key)
            }
            model_names = list(recommendations.keys())
            n_models = len(model_names)
            cols = st.columns(n_models)
            for i, name in enumerate(model_names):
                with cols[i]:
                    st.markdown(f"#### {name}")
                    recs = recommendations[name]
                    if not recs:
                        st.write("— (keine Empfehlungen) —")
                    for rec in recs:
                        if name == "TMDb-Similar":
                            poster_url = rec.get("poster_url", None)
                        else:
                            poster_url = get_tmdb_poster_url(rec["movieId"], links_df, api_key)
                        short_title = rec['title'][:22] + "…" if len(rec['title']) > 24 else rec['title']
                        if poster_url:
                            st.image(poster_url, width=95)
                        else:
                            st.write("—")
                        st.caption(short_title)

                    
# === ADMIN-Bereich ===
if role == "admin":


    # --- Erweiterte Admin-Optionen ---
    st.subheader("⚙️ Pipeline-Optionen")

    # (A) Erzwinge neue Datenverarbeitung
    #force_rebuild = st.checkbox("Neues Dataset (Force rebuild)?", value=False)

    # (B) Test-User-Konfiguration für Validierung
    test_user_count = st.slider("Anzahl Test-User für Validierung", 10, 200, 100)


    # (D) Parameter-Tuning

    n_neighbors = st.slider("KNN Nachbarn (n_neighbors)", 5, 50, 15)
    tfidf_features = st.slider("TF-IDF max_features", 50, 2000, 300)
    latent_dim = st.slider("Latente Dimension (latent_dim)", 8, 128, 32, step=8)
    epochs = st.slider("Epochen (epochs)", 5, 100, 30, step=5)

    # Optionen speichern (z. B. als dict in st.session_state, um sie in der Trigger-API zu übergeben)
    st.session_state["pipeline_conf"] = {
        "test_user_count": test_user_count,
        "n_neighbors": n_neighbors,
        "tfidf_features": tfidf_features,
        "latent_dim": latent_dim,
        "epochs": epochs
    }
    st.subheader("⚙️ Admin Panel – Airflow & MLflow")

    col1, col2  = st.columns(2)

    if "dag_triggered" not in st.session_state:
        st.session_state["dag_triggered"] = False

    with col1:
        # Klassische Pipeline (kein latent_dim!)
        if st.button("▶️ Starte Deep Hybrid Training (DAG: deep_models_pipeline)"):
                try:
                    dag_id = "deep_models_pipeline"
                    airflow_url = API_URL
                    auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

                    deep_conf = {
                        "test_user_count": test_user_count,
                        "n_neighbors": n_neighbors,
                        "latent_dim": latent_dim,
                        "tfidf_features": tfidf_features,
                        "epochs": epochs
                    }

                    # Step 1: Laufende Runs abbrechen
                    running_runs_url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns?state=running"
                    runs_resp = requests.get(running_runs_url, auth=auth)
                    runs_resp.raise_for_status()
                    running_runs = runs_resp.json().get("dag_runs", [])

                    for run in running_runs:
                        run_id = run["dag_run_id"]
                        patch_url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
                        patch_resp = requests.patch(patch_url, json={"state": "failed"}, auth=auth)
                        if patch_resp.status_code == 200:
                            st.info(f"❗ Abgebrochen: Run {run_id}")
                        else:
                            st.warning(f"⚠️ Konnte Run {run_id} nicht abbrechen: {patch_resp.text}")

                    # Step 2: Starte neuen Run
                    response = requests.post(
                        f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns",
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
        if st.button("📊 Zeige Registry-Modelle & Tags (DagsHub)"):
            try:
                from mlflow.tracking import MlflowClient
                import pandas as pd

                client = MlflowClient()
                model_name = "hybrid_deep_model"
                versions = client.search_model_versions(f"name='{model_name}'")

                # Sammle alle Infos in eine Liste
                rows = []
                for v in versions:
                    tags = v.tags
                    row = {
                        "Version": v.version,
                        "Created_at": v.creation_timestamp,
                        "Alias": ", ".join(v.aliases) if hasattr(v, "aliases") else "",
                        "precision_10": tags.get("precision_10", None),
                        "n_neighbors": tags.get("n_neighbors", None),
                        "latent_dim": tags.get("latent_dim", None),
                        "tfidf_features": tags.get("tfidf_features", None),
                        "algorithm": tags.get("algorithm", None),
                    }
                    rows.append(row)
                df = pd.DataFrame(rows)
                # Unix-Timestamp in datetime umwandeln
                if not df.empty:
                    df["Created_at"] = pd.to_datetime(df["Created_at"], unit='ms')
                    df = df.sort_values("Version", ascending=False)
                    st.dataframe(df)

                    # Optional: Chart über die letzten N Versionen
                    if "precision_10" in df.columns:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df["Created_at"],
                            y=pd.to_numeric(df["precision_10"], errors="coerce"),
                            mode="lines+markers",
                            name="Precision@10 Registry",
                            line=dict(color="#E45756", width=2),
                            marker=dict(size=8)
                        ))
                        fig.update_layout(
                            title="Precision@10 je Registry-Version",
                            xaxis_title="Created at",
                            yaxis_title="Precision@10",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Keine Modelle in der Registry gefunden.")
            except Exception as e:
                st.error(f"Fehler beim Laden der Registry-Metriken: {e}")



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

        # Entscheide, welches task_order zu verwenden ist
        task_order = [
            "import_raw_data", "make_dataset", "build_features",
            "train_deep_hybrid_model", "validate_models", "predict_best_model"
        ]


        import time

        for attempt in range(40):
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