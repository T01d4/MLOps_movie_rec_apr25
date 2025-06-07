import streamlit as st
from auth import set_mlflow_from_env
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


def ensure_best_embedding_exists():
    import subprocess
    best_embedding_path = "data/processed/hybrid_deep_embedding_best.csv"
    if not os.path.exists(best_embedding_path):
        try:
            subprocess.run(["dvc", "pull", best_embedding_path], check=True)
            assert os.path.exists(best_embedding_path), "DVC pull hat das File nicht erzeugt!"
        except Exception as e:
            st.error(f"DVC pull fehlgeschlagen für Best-Embedding: {e}")
            raise
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

@st.cache_data(show_spinner="Lade TMDb Similar…")
def get_tmdb_similar_recommendations_cached(selected_movie, movies_df, links_df, api_key):
    import requests
    try:
        row = movies_df[movies_df["title"] == selected_movie]
        if row.empty:
            row = movies_df[movies_df["title"].str.lower() == selected_movie.lower()]
        if not row.empty:
            movie_id = row["movieId"].values[0]
        else:
            movie_id = None
        tmdb_id = None
        if movie_id is not None:
            row = links_df[links_df["movieId"] == movie_id]
            if not row.empty and not pd.isna(row["tmdbId"].values[0]):
                tmdb_id = int(row["tmdbId"].values[0])
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/recommendations"
            params = {"api_key": api_key}
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                results = resp.json().get("results", [])[:10]
                out = []
                for movie in results:
                    poster_url = f"https://image.tmdb.org/t/p/w342{movie['poster_path']}" if movie.get("poster_path") else None
                    out.append({
                        "title": movie["title"],
                        "tmdbId": movie["id"],
                        "poster_url": poster_url
                    })
                return out
        return []
    except Exception:
        return []

def show_best_model_info():
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias("hybrid_deep_model", "best_model")
        version = mv.version
        run_id = mv.run_id
        tags = mv.tags
        st.info(f"🔖 Aktive Modell-Version (`@best_model`): **v{version}** (Run ID: {run_id})")
        keys_of_interest = ["precision_10", "n_neighbors", "latent_dim", "tfidf_features", "algorithm"]
        shown = False
        for k in keys_of_interest:
            if k in tags:
                st.markdown(f"**{k}**: `{tags[k]}`")
                shown = True
        other_tags = {k: v for k, v in tags.items() if k not in keys_of_interest}
        if other_tags:
            with st.expander("Weitere Modell-Tags"):
                for k, v in other_tags.items():
                    st.write(f"{k}: {v}")
        elif not shown:
            st.info("ℹ️ Keine Tags an dieser Modellversion hinterlegt.")
    except Exception as e:
        st.warning(f"Kein best_model-Alias gefunden oder Fehler: {e}")

def get_deep_hybrid_knn_best_recommendations(selected_movies):
    import traceback
    try:
        matrix_path = ensure_best_embedding_exists()
        movies_df = pd.read_csv("data/raw/movies.csv")
        embedding_df = pd.read_csv(matrix_path, index_col=0)
        selected_movie_ids = movies_df[movies_df["title"].str.lower().isin([t.lower() for t in selected_movies])]["movieId"].tolist()
        selected_indices = [embedding_df.index.get_loc(mid) for mid in selected_movie_ids if mid in embedding_df.index]
        if not selected_indices:
            return []
        user_vec = embedding_df.iloc[selected_indices].values.mean(axis=0).reshape(1, -1)
        deep_knn = mlflow.pyfunc.load_model("models:/hybrid_deep_model@best_model")
        user_df = pd.DataFrame(user_vec, columns=embedding_df.columns).astype(np.float32)
        rec_indices = deep_knn.predict(user_df)
        rec_movie_ids = [embedding_df.index[idx] for idx in rec_indices[0] if embedding_df.index[idx] not in selected_movie_ids]
        recommended = movies_df[movies_df["movieId"].isin(rec_movie_ids)].copy()
        recommended = recommended.drop_duplicates("movieId")
        return [{"movieId": int(row["movieId"]), "title": row["title"]} for _, row in recommended.head(10).iterrows()]
    except Exception:
        st.error(traceback.format_exc())
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
    except Exception:
        return []

def get_baseline_recommendations(selected_movies):
    try:
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
    except Exception:
        return []

def show_recommender_ui(user_role="guest"):
    import os
    import requests
    st.header("🎬 Filmempfehlungen")
    try:
        movies_df = pd.read_csv("data/raw/movies.csv")
        movie_titles = sorted(movies_df["title"].dropna().unique())
        data_available = True
    except Exception as e:
        st.warning(f"❌ Filme konnten nicht geladen werden: {e}")
        data_available = False

    if not data_available:
        return

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