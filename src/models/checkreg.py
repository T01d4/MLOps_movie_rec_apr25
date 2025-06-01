# src/models/checkreg.py

import os
from mlflow.tracking import MlflowClient
import logging
from dotenv import load_dotenv

def check_registry():
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MODEL_NAME = "movie_model"
    ALIAS = "best_model"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logging.info(f"🔍 Checking model registry for: '{MODEL_NAME}' and alias '@{ALIAS}'")
    try:
        alias_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        version_num = alias_version.version
        logging.info(f"✅ Alias '@{ALIAS}' zeigt auf Version: {version_num}")
    except Exception as e:
        logging.error(f"❌ Fehler: Alias '@{ALIAS}' nicht gefunden! {e}")
        return False

    model_version = client.get_model_version(MODEL_NAME, version_num)
    logging.info(f"Details Version {version_num}: status={model_version.status}, description={model_version.description}")
    artifact_uri = model_version.source
    logging.info(f"Artifact URI: {artifact_uri}")

    import mlflow.pyfunc
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{ALIAS}")
        logging.info("✅ python_function flavor: Modell kann als pyfunc geladen werden!")
    except Exception as e:
        logging.error(f"❌ Fehler: Modell kann nicht als pyfunc geladen werden ({e})")
        return False

    logging.info("🎉 Model registry check abgeschlossen.")
    return True

if __name__ == "__main__":
    check_registry()