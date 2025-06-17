#auth.py
import streamlit as st
import requests
import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

def show_login():
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    with st.sidebar:
        st.title("🔐 Login")
        username = st.text_input("Benutzername")
        password = st.text_input("Passwort", type="password")
        if st.button("Login"):
            try:
                resp = requests.post(
                    f"{API_URL}/login",
                    data={
                        "username": username,
                        "password": password,
                        "grant_type": "password"
                    },
                    headers={"accept": "application/json"},
                    timeout=6
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["jwt_token"] = data["access_token"]
                    st.session_state["role"] = data["role"]
                    st.session_state["username"] = username
                    st.success(f"Logged in as {data['role'].upper()}")
                    st.rerun()
                else:
                    st.error(f"❌ Login failed – Status: {resp.status_code} – {resp.text}")
            except Exception as e:
                st.error(f"❌ Login API not reachable: {e}")
    jwt_token = st.session_state.get("jwt_token")
    if jwt_token:
        return st.session_state["username"]
    return None

def get_user_role(user):
    return st.session_state.get("role", "guest")


def set_mlflow_from_env():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

def set_airflow():
    API_URL = os.getenv("API_URL")
    AIRFLOW_USER = "admin"
    AIRFLOW_PASS = "admin"


