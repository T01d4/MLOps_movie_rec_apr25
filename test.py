import mlflow
import os
import dotenv
# .env-Datei laden
# dotenv.load_dotenv()
dotenv.load_dotenv(override=True)

print("User:", os.getenv("DAGSHUB_USER"))
print("Pass:", os.getenv("DAGSHUB_TOKEN"))

# os.environ["MLFLOW_TRACKING_USERNAME"] = "DAGSHUB_USERNAME"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "DAGSHUB_TOKEN"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
mlflow.set_tracking_uri("https://dagshub.com/sacer11/MLOps_movie_rec_apr25.mlflow")
mlflow.set_experiment("diagnose_permissions")
try:
    with mlflow.start_run():
        mlflow.log_param("permission_test", "works!")
    print("✅ Test-Log erfolgreich! (Write-Berechtigung ist korrekt.)")
except Exception as e:
    print("❌ Fehler beim Loggen: ", e)