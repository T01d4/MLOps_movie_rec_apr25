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
import json
from mlflow.tracking import MlflowClient


load_dotenv()

API_URL = os.getenv("API_URL", "http://api_service:8000") 

def get_recommendations_via_api(selected_movies):
    api_url = os.getenv("API_URL", "http://api_service:8000")
    endpoint = f"{api_url}/recommend"
    payload = {"selected_movies": selected_movies}
    try:
        resp = requests.post(endpoint, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()  # Dict containing all model outputs
    except Exception as e:
        st.error(f"Error while querying the recommendation API: {e}")
        return {}

def show_best_model_info():
    # Load local config
    config_path = os.path.join(os.getenv("DATA_DIR", "/app/data"), "processed", "pipeline_conf.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias("hybrid_deep_model", "best_model")
        version = mv.version
        run_id = mv.run_id
        tags = mv.tags

        st.info(f"🔖 Active model version (`@best_model`): **v{version}** (Run ID: {run_id})")

        # Mapping between MLflow tags and config keys
        tag_to_json_map = {
            "precision_10": "precision_10",
            "n_neighbors": "n_neighbors",
            "latent_dim": "latent_dim",
            "hidden_dim": "hidden_dim",
            "tfidf_features": "tfidf_features",
            "epochs": "epochs",
            "lr": "lr",
            "batch_size": "batch_size",
            "metric": "metric",
            "content_weight": "content_weight",
            "collab_weight": "collab_weight",
            "power_factor": "power_factor",
            "drop_threshold": "drop_threshold",
        }

        col1, col2 = st.columns(2)
        for i, (tag_key, json_key) in enumerate(tag_to_json_map.items()):
            if tag_key in tags:
                tag_val = tags[tag_key]
                json_val = config.get(json_key)
                if json_val is not None:
                    display = f"`{tag_val}` _(trainiert: {json_val})_"
                else:
                    display = f"`{tag_val}`"
                col = col1 if i % 2 == 0 else col2
                col.markdown(f"**{tag_key}**: {display}")

        # Show other tags
        other_tags = {k: v for k, v in tags.items() if k not in tag_to_json_map}
        if other_tags:
            with st.expander("🧩 Additional model tags", expanded=False):
                for k, v in other_tags.items():
                    st.write(f"**{k}**: `{v}`")
        else:
            st.info("ℹ️ No additional tags found for this model version.")

    except Exception as e:
        st.warning(f"No best_model alias found or error occurred: {e}")

def show_recommender_ui(user_role="guest"):
    st.header("🎬 Movie Recommendations")

    try:
        movies_df = pd.read_csv("data/raw/movies.csv")
        movie_titles = sorted(movies_df["title"].dropna().unique())
        data_available = True
    except Exception as e:
        st.warning(f"❌ Could not load movie list: {e}")
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

    if len(selected_movies) >= 3 and st.button("Recommend 10 Movies"):
        show_best_model_info()
        recommendations = get_recommendations_via_api(selected_movies)
        if not recommendations:
            st.warning("No recommendations received!")
        else:
            model_names = list(recommendations.keys())
            n_models = len(model_names)
            cols = st.columns(n_models)
            for i, name in enumerate(model_names):
                with cols[i]:
                    st.markdown(f"#### {name}")
                    recs = recommendations[name]
                    if not recs:
                        st.write("— (no recommendations) —")
                    for rec in recs:
                        poster_url = rec.get("poster_url", None)
                        short_title = rec['title'][:22] + "…" if len(rec['title']) > 24 else rec['title']
                        if poster_url:
                            st.image(poster_url, width=95)
                        else:
                            st.write("—")
                        st.caption(short_title)