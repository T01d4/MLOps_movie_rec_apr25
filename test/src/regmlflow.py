import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import os
from dotenv import load_dotenv
import logging

# === Logging aktivieren ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Umgebungsvariablen laden ===
load_dotenv(dotenv_path="/opt/airflow/.env")

# === MLflow-Tracking-URI setzen ===
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# === Modellname definieren ===
model_name = "movie_model"

# === Dummy-Modell trainieren ===
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()
model.fit(X, y)

# === Modell loggen und registrieren ===
with mlflow.start_run() as run:
    logging.info(f"📦 Logging Run ID: {run.info.run_id}")

    # Modell speichern
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name=model_name
)

    # Modell registrieren
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name=model_name
    )

    # === Produktion aktivieren ===
    client = mlflow.tracking.MlflowClient()
    version_number = result.version

    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Production",
        archive_existing_versions=True
    )

    logging.info(f"✅ Modell '{model_name}' v{version_number} erfolgreich in 'Production' verschoben.")