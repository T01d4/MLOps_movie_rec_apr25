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

# === Logging ===
logger = logging.getLogger("airflow.task")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# === Umgebungsvariablen ===


DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
REPORT_DIR = os.getenv("REPORT_DIR", "/opt/airflow/reports")
MONITOR_DIR = os.path.join(DATA_DIR, "monitoring")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# === Pfade ===
MONITORING_CONF_PATH = os.path.join(MONITOR_DIR, "monitoring_conf.json")
TRAIN_PATH = os.path.join(MONITOR_DIR, "train.csv")
TEST_PATH = os.path.join(MONITOR_DIR, "test.csv")
DRIFT_HTML_PATH = os.path.join(REPORT_DIR, "drift_report.html")
DRIFT_JSON_PATH = os.path.join(REPORT_DIR, "drift_metrics.json")
PROM_FILE = os.path.join(REPORT_DIR, "drift_metrics.prom")
PLOT_DIR = os.path.join(REPORT_DIR, "plots")
METRICS_PATH = os.path.join(MONITOR_DIR, "metrics_from_mlflow.csv")
LOSS_PATH = os.path.join(REPORT_DIR, "training_loss_per_epoch.json")
#os.makedirs(REPORT_DIR, exist_ok=True)
#os.makedirs(PLOT_DIR, exist_ok=True)
#os.chmod(PLOT_DIR, 0o777)  # oder gezielter: 0o755
# === Konfiguration laden ===
try:
    with open(MONITORING_CONF_PATH, "r") as f:
        conf = json.load(f)
    drift_threshold = conf.get("drift_alert_threshold", 0.3)
    logger.info(f"📥 Konfiguration geladen von {MONITORING_CONF_PATH}")
except Exception as e:
    logger.warning(f"⚠️ Konfiguration konnte nicht geladen werden – Default: {e}")
    drift_threshold = 0.3

# === CSV-Daten laden ===
try:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH).tail(1000)
    logger.info(f"✅ Trainings- und Testdaten geladen")
except Exception as e:
    logger.error(f"❌ Fehler beim Laden der CSV-Dateien: {e}")
    raise
# === Evidently Report erzeugen ===
try:
    column_mapping = ColumnMapping(target="target")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=train, current_data=test, column_mapping=column_mapping)
    temp_output_path = DRIFT_HTML_PATH  + ".tmp"
    report.save_html(temp_output_path)
    os.replace(temp_output_path, DRIFT_HTML_PATH)  # ➜ Atomares Überschreiben

    report.save_html(DRIFT_HTML_PATH)
    report.save_json(DRIFT_JSON_PATH)
    logger.info(f"📊 Evidently Report gespeichert: {DRIFT_HTML_PATH}")
except Exception as e:
    logger.error(f"❌ Fehler beim Erstellen des Reports: {e}")
    raise

# === Prometheus Metriken exportieren ===
def safe_get(metrics, idx, key, default=0.0):
    try:
        return metrics["metrics"][idx]["result"].get(key, default)
    except Exception as e:
        logger.warning(f"⚠️ Zugriff auf metrics[{idx}]['{key}'] fehlgeschlagen: {e}")
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
    logger.info(f"✅ Prometheus-Metriken gespeichert: {PROM_FILE}")
except Exception as e:
    logger.error(f"❌ Fehler beim Schreiben der Drift-Metriken: {e}")

