from fastapi import APIRouter, Body
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

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

router = APIRouter()

def ensure_best_embedding_exists():
    best_embedding_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
    # Prüfe, ob die Embedding-CSV schon lokal vorhanden ist
    if not os.path.exists(best_embedding_path):
        try:
            # Wenn nicht: Lade aus MLflow Registry, passend zum @best_model
            from mlflow.tracking import MlflowClient
            import tempfile
            import shutil

            model_name = "hybrid_deep_model"
            artifact_name = "features/hybrid_deep_embedding.csv"
            client = MlflowClient()
            mv = client.get_model_version_by_alias(model_name, "best_model")
            run_id = mv.run_id
            with tempfile.TemporaryDirectory() as tmpdir:
                local_artifact_path = client.download_artifacts(run_id, artifact_name, tmpdir)
                assert os.path.exists(local_artifact_path), "MLflow download hat das File nicht erzeugt!"
                shutil.copy(local_artifact_path, best_embedding_path)
        except Exception as e:
            raise RuntimeError(f"MLflow Artifact Download fehlgeschlagen: {e}")
    # Jetzt ist das File garantiert da!
    return best_embedding_path


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

@router.post("/recommend")
def recommend_movies(payload: dict = Body(...)):
    selected_movies = payload.get("selected_movies", [])
    api_key = os.getenv("TMDB_API_KEY")
    movies_df = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    links_df = pd.read_csv(os.path.join(RAW_DIR, "links.csv"))

    result = {}

    # 1. Deep Hybrid-KNN_best
    try:
        matrix_path = ensure_best_embedding_exists()
        embedding_df = pd.read_csv(matrix_path, index_col=0)
        # Lade IMMER das aktuelle best_model aus der MLflow Registry
        deep_knn = mlflow.pyfunc.load_model("models:/hybrid_deep_model@best_model")
        selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
        selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]
        rec = []
        if selected_indices:
            user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            user_df = pd.DataFrame(user_vec, columns=embedding_df.columns).astype(np.float32)
            rec_indices = deep_knn.predict(user_df)
            rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0] if embedding_df.index[idx] not in selected_movie_ids]
            recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            rec = [{"movieId": int(row["movieId"]), "title": row["title"], "poster_url": get_tmdb_poster_url(row["movieId"], links_df, api_key)} for _, row in recommended.head(10).iterrows()]
        result["Deep Hybrid-KNN_best"] = rec
    except Exception as e:
        result["Deep Hybrid-KNN_best"] = []

    # 2. Deep Hybrid-KNN_local
    try:
        matrix_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
        model_path = os.path.join(MODELS_DIR, "hybrid_deep_knn.pkl")
        embedding_df = pd.read_csv(matrix_path, index_col=0)
        with open(model_path, "rb") as f:
            deep_knn = pickle.load(f)
        selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
        selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]
        rec = []
        if selected_indices:
            user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
            _, rec_indices = deep_knn.kneighbors(user_vec, n_neighbors=10 + len(selected_indices))
            rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0] if embedding_df.index[idx] not in selected_movie_ids]
            recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
            recommended = recommended.drop_duplicates("movieId")
            rec = [{"movieId": int(row["movieId"]), "title": row["title"], "poster_url": get_tmdb_poster_url(row["movieId"], links_df, api_key)} for _, row in recommended.head(10).iterrows()]
        result["Deep Hybrid-KNN_local"] = rec
    except Exception:
        result["Deep Hybrid-KNN_local"] = []

    # 3. Basis Modell
    try:
        tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv")).dropna(subset=["tag"])
        scores = pd.read_csv(os.path.join(RAW_DIR, "genome-scores.csv"))
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

    return result