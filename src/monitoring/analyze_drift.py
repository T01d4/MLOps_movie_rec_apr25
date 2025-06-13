# ===: monitoring/analyze_drift.py ===
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

# ===: Konfiguration & Pfade
DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
REPORT_DIR = os.environ.get("REPORT_DIR", "/opt/airflow/reports")
MONITORING_CONF_PATH = os.path.join(DATA_DIR, "monitoring", "monitoring_conf.json")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "monitoring")
TRAIN_PATH = os.path.join(CLEAN_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(CLEAN_DATA_DIR, "test.csv")
DRIFT_JSON_PATH = os.path.join(REPORT_DIR, "drift_metrics.json")
DRIFT_HTML_PATH = os.path.join(REPORT_DIR, "drift_report.html")
PROM_FILE = os.path.join(REPORT_DIR, "drift_metrics.prom")

os.makedirs(REPORT_DIR, exist_ok=True)

# ===: Monitoring-Konfiguration laden
try:
    with open(MONITORING_CONF_PATH, "r") as f:
        conf = json.load(f)
    logging.info(f"📂 Konfiguration geladen von {MONITORING_CONF_PATH}")
except Exception as e:
    logging.error(f"❌ Fehler beim Laden der Monitoring-Konfiguration: {e}")
    raise

precision_target = conf.get("precision_target", 0.8)
drift_alert_threshold = conf.get("drift_alert_threshold", 0.3)
latency_alert_threshold = conf.get("latency_alert_threshold", 0.5)

# ===: Daten laden
try:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    logging.info(f"✅ Trainings- und Testdaten geladen ({TRAIN_PATH}, {TEST_PATH})")
except Exception as e:
    logging.error(f"❌ Fehler beim Laden der CSV-Dateien: {e}")
    raise

# ===: Report erstellen
try:
    column_mapping = ColumnMapping()
    column_mapping.target = "target"  # ggf. anpassen

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=train, current_data=test, column_mapping=column_mapping)
    report.save_html(DRIFT_HTML_PATH)
    report.save_json(DRIFT_JSON_PATH)
    logging.info(f"📊 Evidently-Bericht gespeichert: {DRIFT_HTML_PATH}")
except Exception as e:
    logging.error(f"❌ Fehler beim Erstellen des Drift-Reports: {e}")
    raise

# ===: JSON einlesen und Prometheus-Metriken schreiben
try:
    with open(DRIFT_JSON_PATH, "r") as f:
        drift_data = json.load(f)
    logging.info("✅ drift_metrics.json loaded.")
    metrics = {
        "data_drift_share": drift_data["metrics"][0]["result"].get("dataset_drift", 0.0),
        "target_drift_psi": drift_data["metrics"][1]["result"].get("psi", 0.0),
        "n_drifted_columns": drift_data["metrics"][0]["result"].get("number_of_drifted_columns", 0),
        "share_drifted_columns": drift_data["metrics"][0]["result"].get("share_of_drifted_columns", 0.0),
    }

    with open(PROM_FILE, "w") as f:
        f.write(f"data_drift_share {metrics['data_drift_share']:.4f}\n")
        f.write(f"target_drift_psi {metrics['target_drift_psi']:.4f}\n")
        f.write(f"drifted_columns_total {metrics['n_drifted_columns']}\n")
        f.write(f"drifted_columns_share {metrics['share_drifted_columns']:.4f}\n")

    logging.info(f"✅ Prometheus-Metriken gespeichert unter: {PROM_FILE}")
except Exception as e:
    logging.error(f"❌ Fehler beim Parsen oder Schreiben der Drift-Metriken: {e}")
    raise