#app.py
import streamlit as st

from auth import show_login, get_user_role
from auth import set_mlflow_from_env
from recommender import show_recommender_ui
from training import show_admin_panel
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import logging
import mlflow
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

load_dotenv(".env")
set_mlflow_from_env()

required_env_vars = [
    "MLFLOW_TRACKING_URI", "DAGSHUB_USER",
    "DAGSHUB_TOKEN", "API_URL"
]

missing = [v for v in required_env_vars if os.getenv(v) is None]
if missing:
    st.error(f"❌ Fehlende .env Einträge: {missing}")
    st.stop()


st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

user = show_login()
if not user:
    st.stop()

role = get_user_role(user)
st.success(f"Angemeldet als: {role.upper()}")

if role == "admin":
    show_admin_panel()
    st.markdown("---")
    show_recommender_ui(user_role="admin")
elif role in ["user", "guest"]:
    show_recommender_ui(user_role=role)
else:
    st.warning("Unbekannte Rolle. Bitte neu einloggen.")
    st.stop()