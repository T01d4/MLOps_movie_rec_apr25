# src/monitoring/analyze_drift.py (erweitert & vereinigt)

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from prometheus_client import Gauge, CollectorRegistry, write_to_textfile
import subprocess
import getpass
import json
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", os.getenv("MLFLOW_TRACKING_URI")))
mlflow.set_experiment("hybrid_deep_model")

registry = CollectorRegistry()

# === Logging ===
logger = logging.getLogger("airflow.task")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# === Environment variables ===


DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
REPORT_DIR = os.getenv("REPORT_DIR", "/opt/airflow/reports")
MONITOR_DIR = os.path.join(DATA_DIR, "monitoring")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# === Paths ===
MONITORING_CONF_PATH = os.path.join(MONITOR_DIR, "monitoring_conf.json")
TRAIN_PATH = os.path.join(MONITOR_DIR, "train.csv")
TEST_PATH = os.path.join(MONITOR_DIR, "test.csv")
DRIFT_HTML_PATH = os.path.join(REPORT_DIR, "drift_report.html")
DRIFT_JSON_PATH = os.path.join(REPORT_DIR, "drift_metrics.json")
PROM_FILE = os.path.join(REPORT_DIR, "drift_metrics.prom")
PLOT_DIR = os.path.join(REPORT_DIR, "plots")
METRICS_PATH = os.path.join(MONITOR_DIR, "metrics_from_mlflow.csv")
LOSS_PATH = os.path.join(REPORT_DIR, "training_loss_per_epoch.json")
EMBEDDING_PATH = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
BEST_EMBEDDING_PATH = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding_best.csv")



def log_monitoring_metrics(metrics: dict):
    try:
        client = MlflowClient()
        model_name = "hybrid_deep_model"

        # 1️⃣ Modellversion holen
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
            train_run_id = latest_version.run_id
            logger.info(f"📡 Logging metrics to MODEL VERSION {latest_version.version} / Run: {train_run_id}")
        else:
            logger.warning("⚠️ No model versions found – trying fallback to training run.")
            runs = mlflow.search_runs(
                experiment_names=["hybrid_deep_model"],
                filter_string="tags.task = 'train_hybrid_deep_model'",
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs.empty:
                logger.error("❌ No fallback training run found – cannot log metrics.")
                return
            train_run_id = runs.iloc[0]["run_id"]
            logger.info(f"📡 Logging metrics to TRAINING RUN: {train_run_id}")

        # 2️⃣ Logging
        for key, value in metrics.items():
            if value is not None:
                client.log_metric(run_id=train_run_id, key=key, value=value)
                logger.info(f"✅ Logged {key}: {value}")

    except Exception as e:
        logger.error(f"❌ Failed to log monitoring metrics to MLflow: {e}", exc_info=True)


        
# === Load configuration ===
try:
    with open(MONITORING_CONF_PATH, "r") as f:
        conf = json.load(f)
    drift_threshold = conf.get("drift_alert_threshold", 0.3)
    logger.info(f"📥 Configuration loaded from {MONITORING_CONF_PATH}")
except Exception as e:
    logger.warning(f"⚠️ Failed to load configuration – using default – Default: {e}")
    drift_threshold = 0.3

# === Load CSV data ===
try:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH).tail(1000)
    logger.info(f"✅ Training and test data loaded")
except Exception as e:
    logger.error(f"❌ Error loading CSV files: {e}")
    raise
# === Generate Evidently Report ===
try:
    column_mapping = ColumnMapping(target="target")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=train, current_data=test, column_mapping=column_mapping)
    temp_output_path = DRIFT_HTML_PATH  + ".tmp"
    report.save_html(temp_output_path)
    os.replace(temp_output_path, DRIFT_HTML_PATH)   # ➜ atomic overwrite if blocked by streamlit

    report.save_html(DRIFT_HTML_PATH)
    report.save_json(DRIFT_JSON_PATH)
    logger.info(f"📊 Evidently report saved: {DRIFT_HTML_PATH}")
except Exception as e:
    logger.error(f"❌ Error creating report: {e}")
    raise

# === Export Prometheus metrics ===
def safe_get(metrics, idx, key, default=0.0):
    try:
        return metrics["metrics"][idx]["result"].get(key, default)
    except Exception as e:
        logger.warning(f"⚠️ Failed to access metrics[{idx}]['{key}'] : {e}")
        return default

try:
    with open(DRIFT_JSON_PATH, "r", encoding="utf-8") as f:
        drift_data = json.load(f)

    data_drift = safe_get(drift_data, 0, "dataset_drift")
    target_psi = safe_get(drift_data, 1, "psi")
    n_drifted = safe_get(drift_data, 0, "number_of_drifted_columns", 0)
    share_drift = safe_get(drift_data, 0, "share_of_drifted_columns", 0.0)
    drift_alert = int(data_drift > drift_threshold)

    lines = [
        f"data_drift_share {data_drift:.4f}",
        f"target_drift_psi {target_psi:.4f}",
        f"drifted_columns_total {int(n_drifted)}",
        f"drifted_columns_share {share_drift:.4f}",
        f"drift_alert {drift_alert}"
    ]

    with open(PROM_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"✅ Prometheus metrics saved: {PROM_FILE}")
except Exception as e:
    logger.error(f"❌ Error writing drift metrics: {e}")



# === Drift Detection & Prometheus Monitoring (modular) ===


# 2. Extract common columns
try:
    reference_df = pd.read_csv(BEST_EMBEDDING_PATH) 
    current_df = pd.read_csv(EMBEDDING_PATH)
    logger.info("📥 Embedding files successfully loaded.")
except Exception as e:
    logger.error(f"❌ Error loading embedding files: {e}")
    raise

# 2. Extract common columns
try:
    common_cols = sorted(set(reference_df.columns) & set(current_df.columns))
    if not common_cols:
        raise ValueError("⚠️ No common columns found for drift analysis.")
    reference_df = reference_df[common_cols]
    current_df = current_df[common_cols]
    logger.info(f"✅ {len(common_cols)} common columns for drift analysis.")
except Exception as e:
    logger.error(f"❌ Error selecting columns: {e}")
    raise

# 3. First Evidently Report for Prometheus .prom export
try:
    column_mapping = ColumnMapping(numerical_features=common_cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    drift_json = report.as_dict()
    drift_alert = int(drift_json["metrics"][0]["result"]["dataset_drift"])
    drift_share = drift_json["metrics"][0]["result"]["share_of_drifted_columns"]

    drift_file = os.path.join(REPORT_DIR, "training_metrics.prom")
    with open(drift_file, "w") as f:
        f.write(f'model_drift_alert{{model="Deep Hybrid-KNN_best"}} {drift_alert}\n')
        f.write(f'data_drift_share{{model="Deep Hybrid-KNN_best"}} {drift_share:.4f}\n')

    logger.info("📈 Classic Prometheus metrics saved.")
except Exception as e:
    logger.error(f"❌ Error generating classic drift metrics: {e}")

# 4. Second Evidently report with robust drift_score (for time series analysis)
try:
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    drift_result = drift_report.as_dict()

    drift_score = None
    for m in drift_result.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            drift_score = m.get("result", {}).get("drift_share", None)
            break

    drift_alert = int(drift_score > drift_threshold) if drift_score is not None else 0



  # Prometheus labeled metrics
    DRIFT_ALERT = Gauge("drift_alert", "Drift alert flag", ["model"], registry=registry)
    DRIFT_ALERT.labels(model="hybrid_deep_model").set(drift_alert)

    DRIFT_SCORE = Gauge("drift_score_sliding_window", "Drift score over recent requests", ["model"], registry=registry)
    if drift_score is not None:
        DRIFT_SCORE.labels(model="hybrid_deep_model").set(drift_score)

    # === Log Metrics in MLflow ===
    monitoring_metrics = {
        "drift_score_sliding_window": drift_score,
        "model_drift_alert": drift_alert,
        "target_drift_psi": target_psi,
        "drifted_columns_share": share_drift,
        "drifted_columns_total": n_drifted
    }
    log_monitoring_metrics(monitoring_metrics)

    DRIFT_METRICS_PATH = os.path.join(REPORT_DIR, "drift_score_sliding.prom")
    write_to_textfile(DRIFT_METRICS_PATH, registry)

    if drift_score is not None:
        logger.info(f"🔍  Drift detected in {drift_score:.2%} of features (Alert={drift_alert})")
    else:
        logger.warning("⚠️ Drift share could not be determined (None).")
        logger.info(f"Ref shape: {reference_df.shape}, Cur shape: {current_df.shape}")
except Exception as e:
    logger.error(f"❌ Error during sliding drift export:: {e}")

# 5. JSON-Export
try:
    json_path = os.path.join(REPORT_DIR, "drift_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drift_result, f, indent=2)
    logger.info(f"📄 Drift metrics saved as JSON: {json_path}")
except Exception as e:
    logger.error(f"❌ Error saving drift metrics as JSON: {e}")



