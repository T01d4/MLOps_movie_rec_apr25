from fastapi import APIRouter, Body
from fastapi.responses import Response
import pandas as pd
import numpy as np
import mlflow
import os
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import json
import tempfile
import shutil
from prometheus_client import Counter, Summary, Gauge, Histogram, generate_latest
import time
import datetime

load_dotenv()

router = APIRouter()
CONVERSIONS = Counter("recommendation_conversions_total", "Clicked recommended item")
#CONVERSION_RATE = Gauge("conversion_rate", "Click/view rate per model", ["model"])
RECOMMENDATIONS = Counter("recommendation_shown_total", "Total recommendations shown")
REQUEST_COUNT = Counter("recommendation_requests_total", "Anzahl der Recommendation-Requests")
CONVERSION_RATE = Gauge("conversion_rate", "Click-through rate on recommendations")
# Jetzt mit "model"-Label
REQUEST_LATENCY = Summary("recommendation_latency_seconds", "Latenz der Vorhersage in Sekunden", ["model"])
API_USAGE = Counter("recommendation_api_usage_total", "API-Interaktionen", ["model"])
DRIFT_ALERT = Gauge("drift_alert", "Drift Alert Status (1=drift, 0=ok)", ["model"])
REQUEST_LOG_PATH = os.path.join(os.getenv("DATA_DIR", "data"), "monitoring", "api_requests.csv")
DRIFT_LOG_PATH = os.getenv("DRIFT_LOG_PATH", "/app/data/drift_request_log.csv")
CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence", "Approximated prediction confidence", buckets=[i/10 for i in range(11)]
)
os.makedirs(os.path.dirname(REQUEST_LOG_PATH), exist_ok=True)


def ensure_best_embedding_exists():
    best_embedding_path = "data/processed/hybrid_deep_embedding_best.csv"
    # Check if the embedding CSV is already available locally
    if not os.path.exists(best_embedding_path):
        try:
            # If not: download from MLflow registry for @best_model
            model_name = "hybrid_deep_model"
            artifact_name = "best_embedding/hybrid_deep_embedding_best.csv"
            client = MlflowClient()
            mv = client.get_model_version_by_alias(model_name, "best_model")
            run_id = mv.run_id
            with tempfile.TemporaryDirectory() as tmpdir:
                local_artifact_path = client.download_artifacts(run_id, artifact_name, tmpdir)
                assert os.path.exists(local_artifact_path), "MLflow download hat das File nicht erzeugt!"
                shutil.copy(local_artifact_path, best_embedding_path)
        except Exception as e:
            raise RuntimeError(f"MLflow artifact download failed: {e}")
    else:
        # Falls lokal vorhanden, lade Version trotzdem
        try:
            client = MlflowClient()
            mv = client.get_model_version_by_alias("hybrid_deep_model", "best_model")
        except:
            pass  # Fallback
    # At this point the file is guaranteed to exist
    return best_embedding_path


def load_config_from_best_model():

    client = MlflowClient()
    mv = client.get_model_version_by_alias("hybrid_deep_model", "best_model")
    run_id = mv.run_id
    config_path = client.download_artifacts(run_id, "best_config/pipeline_conf_best.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    except Exception:
        return None

def log_embedding_for_drift(embedding_vector: np.ndarray):
    timestamp = datetime.datetime.utcnow().isoformat()
    df = pd.DataFrame([embedding_vector], columns=[f"f{i}" for i in range(len(embedding_vector))])
    df["timestamp"] = timestamp
    os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
    header = not os.path.exists(DRIFT_LOG_PATH)
    df.to_csv(DRIFT_LOG_PATH, mode="a", header=header, index=False)

@router.get("/metrics/training")
def custom_metrics():
    return Response(generate_latest(), media_type="text/plain")

@router.get("/recommend")
def recommend_get_warning():
    return {"error": "Use POST request with selected_movies payload"}

@router.post("/track_click")
def track_click(item_id: int):
    CONVERSIONS.inc()
    return {"message": "conversion logged"}

@router.post("/recommend")
def recommend_movies(payload: dict = Body(...)):
    start_time = time.time()
    REQUEST_COUNT.inc()

        # read Drift-Alert from Evidently 
    try:
        with open("reports/drift_metrics.json", "r", encoding="utf-8") as f:
            drift_data = json.load(f)
        drift_score = drift_data["metrics"][0]["result"].get("dataset_drift", 0.0)
        DRIFT_ALERT.labels("Deep Hybrid-KNN_best").set(int(drift_score > 0.3))
        DRIFT_ALERT.labels("Deep Hybrid-KNN_local").set(int(drift_score > 0.3))
    except Exception:
        DRIFT_ALERT.set(0)  # Fallback


    selected_movies = payload.get("selected_movies", [])
    api_key = os.getenv("TMDB_API_KEY")
    movies_df = pd.read_csv("data/raw/movies.csv")
    links_df = pd.read_csv("data/raw/links.csv")

    result = {}

    # 1. Deep Hybrid-KNN_best
    try:
        matrix_path = ensure_best_embedding_exists()
        embedding_df = pd.read_csv(matrix_path, index_col=0)
        embedding_df.index = embedding_df.index.astype(int)
       
        config = load_config_from_best_model()
        print("Loaded best config:", config)
        # Always load the current best_model from MLflow registry
        deep_knn = mlflow.pyfunc.load_model("models:/hybrid_deep_model@best_model")
        selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
        selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]
        rec = []
        
        if not selected_indices:
            print("⚠️ Keine gültigen Movie-IDs in embedding_df gefunden.")
            result["Deep Hybrid-KNN_best"] = []
            return result

        if selected_indices:

            
            user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            conf_score = 0.5  # mittlere Sicherheit als Fallback
            CONFIDENCE_HISTOGRAM.observe(conf_score)
            log_embedding_for_drift(user_vec.flatten())
            user_df = pd.DataFrame(user_vec, columns=embedding_df.columns).astype(np.float32)
            user_df["timestamp"] = datetime.datetime.now().isoformat()

            try:
                if not os.path.exists(REQUEST_LOG_PATH):
                    user_df.to_csv(REQUEST_LOG_PATH, index=False)
                else:
                    user_df.to_csv(REQUEST_LOG_PATH, mode="a", header=False, index=False)
                print(f"📥 Nutzer-Embedding geloggt: {user_df.shape}")
                
            except Exception as e:
                print(f"⚠️ Logging fehlgeschlagen: {e}")
            


            rec_indices = deep_knn.predict(user_df)
            rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0] if embedding_df.index[idx] not in selected_movie_ids]
            print("🔍 Rec indices:", rec_indices)
            print("✅ Selected movie ids:", selected_movie_ids)
            print("✅ Recomm. movie ids:", rec_movie_ids)
            recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            rec = [{"movieId": int(row["movieId"]), "title": row["title"], "poster_url": get_tmdb_poster_url(row["movieId"], links_df, api_key)} for _, row in recommended.head(10).iterrows()]     
        result["Deep Hybrid-KNN_best"] = rec
        API_USAGE.labels(model="Deep Hybrid-KNN_best").inc()
        REQUEST_LATENCY.labels(model="hybrid_deep_model").observe(time.time() - start_time)
    except Exception as e:
        result["Deep Hybrid-KNN_best"] = []

    # 2. Deep Hybrid-KNN_local
    try:
        matrix_path = "data/processed/hybrid_deep_embedding.csv"
        model_path = "models/hybrid_deep_knn.pkl"
        config_path = "data/processed/pipeline_conf.json"

        embedding_df = pd.read_csv(matrix_path, index_col=0)
        embedding_df.index = embedding_df.index.astype(int)
        with open(model_path, "rb") as f:
            deep_knn = pickle.load(f)

        # Load local pipeline_conf.json
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
            print(f"⚠️ No pipeline_conf.json found at {config_path}")

        selected_movie_ids = movies_df[movies_df["title"].str.lower().isin(
            [t.lower() for t in selected_movies])]["movieId"].tolist()
        selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]

        rec = []
        if selected_indices:
            user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            _, rec_indices = deep_knn.kneighbors(user_vec, n_neighbors=10 + len(selected_indices))
            distances, rec_indices = deep_knn.kneighbors(user_vec, n_neighbors=10 + len(selected_indices))
            conf_score = 1 - np.mean(distances)
            CONFIDENCE_HISTOGRAM.observe(conf_score)
            rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0]
                             if embedding_df.index[idx] not in selected_movie_ids]
            recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            rec = [{
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "poster_url": get_tmdb_poster_url(row["movieId"], links_df, api_key)
            } for _, row in recommended.head(10).iterrows()]

        result["Deep Hybrid-KNN_local"] = rec
        API_USAGE.labels(model="Deep Hybrid-KNN_local").inc()
        REQUEST_LATENCY.labels(model="Deep Hybrid-KNN_local").observe(time.time() - start_time)
    except Exception as e:
        print(f"❌ Error during local prediction: {e}")
        result["Deep Hybrid-KNN_local"] = []

    # 3. Basis Modell
    try:
        tags = pd.read_csv("data/raw/tags.csv").dropna(subset=["tag"])
        scores = pd.read_csv("data/raw/genome-scores.csv")
        tags_combined = tags.groupby("movieId")["tag"].apply(lambda t: " ".join(t)).reset_index()
        movies = pd.merge(movies_df, tags_combined, on="movieId", how="left")
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
        rec = []
        if selected_indices:
            knn = NearestNeighbors(n_neighbors=10 + len(selected_indices), metric="cosine")
            knn.fit(hybrid_df.values)
            user_vec = hybrid_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            _, distances = knn.kneighbors(user_vec)
            _, rec_indices = knn.kneighbors(user_vec)
            rec_movie_ids = [hybrid_df.index[idx] for idx in rec_indices[0] if hybrid_df.index[idx] not in selected_movie_ids]
            recommended = valid_movies[valid_movies["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            rec = [{"movieId": int(row["movieId"]), "title": row["title"], "poster_url": get_tmdb_poster_url(row["movieId"], links_df, api_key)} for _, row in recommended.head(10).iterrows()]
        result["Basis Modell"] = rec
    except Exception:
        result["Basis Modell"] = []

    # 4. TMDb-Similar
    try:
        if selected_movies:
            row = movies_df[movies_df["title"] == selected_movies[0]]
            if row.empty:
                row = movies_df[movies_df["title"].str.lower() == selected_movies[0].lower()]
            if not row.empty:
                movie_id = row["movieId"].values[0]
            else:
                movie_id = None
            tmdb_id = None
            if movie_id is not None:
                row = links_df[links_df["movieId"] == movie_id]
                if not row.empty and not pd.isna(row["tmdbId"].values[0]):
                    tmdb_id = int(row["tmdbId"].values[0])
            rec = []
            if tmdb_id:
                url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/recommendations"
                params = {"api_key": api_key}
                resp = requests.get(url, params=params, timeout=8)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])[:10]
                    for movie in results:
                        poster_url = f"https://image.tmdb.org/t/p/w342{movie['poster_path']}" if movie.get("poster_path") else None
                        rec.append({
                            "title": movie["title"],
                            "tmdbId": movie["id"],
                            "poster_url": poster_url
                        })
            result["TMDb-Similar"] = rec
        else:
            result["TMDb-Similar"] = []
    except Exception:
        result["TMDb-Similar"] = []
    conf_score = 1 - np.mean(distances)  # Annahme: kleinere Distanz = höhere Sicherheit
    CONFIDENCE_HISTOGRAM.observe(conf_score)
    RECOMMENDATIONS.inc()
    try:
        conversion_rate = CONVERSIONS._value.get() / RECOMMENDATIONS._value.get() if RECOMMENDATIONS._value.get() > 0 else 0
        #CONVERSION_RATE.labels(model="Deep Hybrid-KNN_best").set(conversion_rate)
        CONVERSION_RATE.set(conversion_rate)
    except Exception as e:
        print(f"⚠️ Conversion Rate konnte nicht berechnet werden: {e}")
        # Nur wenn der erste (beste) Modell-Pfad verwendet wurde
    return result