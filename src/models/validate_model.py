# src/models/validate_model.py
import pickle
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
import os
import logging
from mlflow.tracking import MlflowClient
import time
import dagshub
# === Logging für Airflow aktivieren ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Setup ===
load_dotenv()
dagshub.init(repo_owner='sacer11', repo_name='MLOps_movie_rec_apr25', mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("model_validation")

def validate_models():
    logging.info("🚀 Starte Validierung der Modelle")
    try:
        ratings = pd.read_csv("/opt/airflow/data/raw/ratings.csv")
        movie_matrix = pd.read_csv("/opt/airflow/data/processed/movie_matrix.csv", index_col=0)
        hybrid_matrix = pd.read_csv("/opt/airflow/data/processed/hybrid_matrix.csv")
        user_matrix = pd.read_csv("/opt/airflow/data/processed/user_matrix.csv", index_col=0)

        with open("/opt/airflow/models/hybrid_model.pkl", "rb") as f:
            hybrid_model = pickle.load(f)
        with open("/opt/airflow/models/user_model.pkl", "rb") as f:
            user_model = pickle.load(f)

        logging.info("📥 Daten geladen – Beginne Evaluation")
    except Exception as e:
        logging.error(f"❌ Fehler beim Laden der Daten oder Modelle: {e}")
        return

    test_users = user_matrix.index[:100]
    hybrid_scores, user_scores, valid_users = [], [], []

    for uid in test_users:
        try:
            uvec_user = user_matrix.loc[uid].values.reshape(1, -1)
            uvec_hybrid = hybrid_matrix.iloc[uid].values.reshape(1, -1)

            if uvec_user.shape[1] != user_model.n_features_in_:
                raise ValueError(f"user_model erwartet {user_model.n_features_in_} Features, hat aber {uvec_user.shape[1]}")
            if uvec_hybrid.shape[1] != hybrid_model.n_features_in_:
                raise ValueError(f"hybrid_model erwartet {hybrid_model.n_features_in_} Features, hat aber {uvec_hybrid.shape[1]}")

            _, idxs_hybrid = hybrid_model.kneighbors(uvec_hybrid)
            hit_hybrid = ratings[(ratings["userId"] == uid) &
                                 (ratings["movieId"].isin(movie_matrix.index[idxs_hybrid[0]]))]
            hybrid_scores.append(1 if not hit_hybrid.empty else 0)

            _, idxs_user = user_model.kneighbors(uvec_user)
            hit_user = ratings[(ratings["userId"] == uid) &
                               (ratings["movieId"].isin(user_matrix.index[idxs_user[0]]))]
            user_scores.append(1 if not hit_user.empty else 0)

            valid_users.append(uid)
        except Exception as e:
            logging.warning(f"⚠️ Fehler bei User {uid}: {e}")
            continue

    if len(valid_users) == 0:
        logging.error("❌ Keine gültigen Nutzer zur Auswertung!")
        return

    hybrid_mean = float(np.mean(hybrid_scores))
    user_mean = float(np.mean(user_scores))

    logging.info(f"📊 precision_10_hybrid: {hybrid_mean:.2f}")
    logging.info(f"📊 precision_10_user:   {user_mean:.2f}")

    try:
        logging.info("📡 Logging der Metriken in MLflow")
        with mlflow.start_run(run_name="validate_predictions") as run:
            mlflow.set_tag("source", "airflow")
            mlflow.set_tag("task", "validate_models")
            mlflow.log_param("n_test_users", len(valid_users))
            mlflow.log_metric("precision_10_hybrid", hybrid_mean)
            mlflow.log_metric("precision_10_user", user_mean)

            score_df = pd.DataFrame({
                "user_id": valid_users,
                "hybrid_score": hybrid_scores,
                "user_score": user_scores
            })
            score_path = "/opt/airflow/data/processed/validation_scores.csv"
            score_df.to_csv(score_path, index=False)
            mlflow.log_artifact(score_path, artifact_path="validation")

            # 📦 MLflow-Modell-Registry aktualisieren
            if hybrid_mean > user_mean:
                pyfunc_model_uri = f"runs:/{run.info.run_id}/hybrid_knn_pyfunc"
                typ = "hybrid"
                prec = f"{hybrid_mean:.2f}"
                logging.info("🏆 Bestes Modell: hybrid_model")
            else:
                pyfunc_model_uri = f"runs:/{run.info.run_id}/user_knn_pyfunc"
                typ = "user"
                prec = f"{user_mean:.2f}"
                logging.info("🏆 Bestes Modell: user_model")

            result = mlflow.register_model(model_uri=pyfunc_model_uri, name="movie_model")
            client = MlflowClient()
            model_name = "movie_model"

            # Typ und Wert ausgeben (im Airflow Log sichtbar!)
            logging.info(f"TYPE of version: {type(result.version)}, VALUE: {result.version}")

            version = result.version
            if not isinstance(version, int):
                try:
                    version = int(str(version))
                except Exception as e:
                    logging.error(f"❌ Modell-Version-Typ konnte nicht in int umgewandelt werden: {e}")
                    return

            # Warte, bis das Modell wirklich bereit ist
            for i in range(20):
                mv = client.get_model_version(model_name, version)
                logging.info(f"Status von Modell-Version {version}: {mv.status}")
                if mv.status == "READY":
                    break
                if mv.status == "FAILED":
                    logging.error(f"❌ Modell-Version {version} ist fehlgeschlagen!")
                    return
                time.sleep(2)
            else:
                logging.error(f"❌ Modell-Version {version} wurde nicht READY. Aktueller Status: {mv.status}")
                return

            # Setze Alias @best_model auf diese Version
            try:
                client.set_registered_model_alias(model_name, "best_model", int(version))
                logging.info(f"✅ Alias @best_model auf Version {version} gesetzt.")
            except Exception as e:
                logging.error(f"❌ Alias konnte nicht gesetzt werden: {e}")

            # Setze Registry-Tags
            try:
                client.set_model_version_tag(model_name, int(version), "type", typ)
                client.set_model_version_tag(model_name, int(version), "precision", prec)
                client.set_model_version_tag(model_name, int(version), "source", "airflow_dag_v1")
                logging.info(f"🏷️ Tags für Typ, Precision und Source gesetzt.")
            except Exception as e:
                logging.error(f"❌ Konnte Tags nicht setzen: {e}")

            # Optional: Beschreibung für die neue Modellversion setzen
            try:
                beschr = (
                    f"Auto-selected {typ} KNN model by Airflow DAG. "
                    f"Precision: {hybrid_mean:.2f} (hybrid), {user_mean:.2f} (user)."
                )
                client.update_model_version(
                    name=model_name,
                    version=int(version),
                    description=beschr
                )
                logging.info("📝 Beschreibung für das Modell gesetzt.")
            except Exception as e:
                logging.error(f"❌ Beschreibung konnte nicht gesetzt werden: {e}")

            # Nach Registry/Tags/Alias alles nochmal ausführlich loggen
            try:
                alias_ver = client.get_model_version_by_alias(model_name, "best_model")
                logging.info(f"🔎 Alias @best_model zeigt auf Version: {alias_ver.version}, Source: {alias_ver.source}")
                logging.info(f"🔎 Aktuelle Model Tags: {alias_ver.tags}")
            except Exception as e:
                logging.error(f"❌ Alias-Check nach Setzen fehlgeschlagen: {e}")

    except Exception as e:
        logging.error(f"❌ Fehler beim MLflow Logging oder Modell-Registrierung: {e}")
        return

    logging.info("🎉 Validation & Registry update abgeschlossen.")

if __name__ == "__main__":
    validate_models()