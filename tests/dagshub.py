import mlflow
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "kasparrobert"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f4c84b4c206ff2aeec881a1262e200d625741ee4c"
mlflow.set_tracking_uri("https://dagshub.com/sacer11/MLOps_movie_rec_apr25.mlflow")

try:
    run = mlflow.start_run()
    mlflow.log_metric("test_metric", 1.0)
    mlflow.end_run()
    print("✅ Token funktioniert!")
except Exception as e:
    print("❌ Fehler:", e)