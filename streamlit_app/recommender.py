#recommender.py
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

API_URL = os.getenv("API_URL", "http://api_service:8000") 

def get_recommendations_via_api(selected_movies):
    api_url = os.getenv("API_URL", "http://api_service:8000")
    endpoint = f"{api_url}/recommend"
    payload = {"selected_movies": selected_movies}
    try:
        resp = requests.post(endpoint, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()  # Dict mit allen vier Modell-Outputs
    except Exception as e:
        st.error(f"Fehler beim Abfragen der Empfehlung-API: {e}")
        return {}

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
        recommendations = get_recommendations_via_api(selected_movies)
        if not recommendations:
            st.warning("Keine Empfehlungen erhalten!")
        else:
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
                        poster_url = rec.get("poster_url", None)
                        short_title = rec['title'][:22] + "…" if len(rec['title']) > 24 else rec['title']
                        if poster_url:
                            st.image(poster_url, width=95)
                        else:
                            st.write("—")
                        st.caption(short_title)