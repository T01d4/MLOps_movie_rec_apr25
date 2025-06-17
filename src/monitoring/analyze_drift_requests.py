# src/monitoring/analyze_drift_requests.py

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from prometheus_client import Gauge
import json
import logging

logger = logging.getLogger("airflow.task")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MONITOR_DIR = os.path.join(DATA_DIR, "monitoring")
REPORT_DIR = os.getenv("REPORT_DIR", "/app/reports")
DRIFT_SCORE = Gauge("drift_score_sliding_window", "Drift score over recent requests", ["model"])
#os.makedirs(REPORT_DIR, exist_ok=True)

reference_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding_best.csv")
requests_path = os.path.join(MONITOR_DIR, "api_requests.csv")
drift_json = os.path.join(REPORT_DIR, "request_drift.json")
drift_prom = os.path.join(REPORT_DIR, "request_drift.prom")

# === Lade Referenzdaten ===
try:
    reference_df = pd.read_csv(reference_path, index_col=0)
    if "timestamp" in reference_df.columns:
        reference_df = reference_df.drop(columns=["timestamp"])
    logger.info(f"✅ Referenzdaten geladen: {reference_path}")
except Exception as e:
    logger.error(f"❌ Fehler beim Laden der Referenzdaten: {e}")
    reference_df = pd.DataFrame()
    logger.warning("⚠️ Leerer Referenz-DataFrame wird verwendet.")

# === Lade Request-Daten ===
try:
    request_df = pd.read_csv(requests_path)
    request_df = request_df.drop(columns=["timestamp"], errors="ignore")
    current_df = request_df.tail(1000).copy()
    logger.info(f"✅ Nutzerdaten geladen, Zeilen: {len(current_df)}")
except Exception as e:
    logger.warning(f"⚠️ Fehler beim Laden von Requests, nutze Referenz als Fallback: {e}")
    current_df = reference_df.copy()

# === Validierung & Report ===
try:
    if reference_df.empty or current_df.empty:
        logger.warning("⚠️ Mindestens einer der DataFrames ist leer – Report wird nicht sinnvoll.")
    else:
        # Gleiche Spaltenreihenfolge & Inhalt sicherstellen
        common_cols = list(set(reference_df.columns) & set(current_df.columns))
        reference_df = reference_df[common_cols]
        current_df = current_df[common_cols]

        # Report ausführen
        column_mapping = ColumnMapping(numerical_features=common_cols)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
        report.save_json(drift_json)
        logger.info(f"📊 Evidently Drift-Report gespeichert: {drift_json}")
except Exception as e:
    logger.error(f"❌ Fehler beim Erzeugen des Evidently-Reports: {e}")

# Prometheus-Metriken schreiben
try:
    result = report.as_dict()
    drift_flag = int(result["metrics"][0]["result"]["dataset_drift"])
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    drift_score = result["metrics"][0]["result"].get("share_of_drifted_columns", 0.0)

    model_name = "Deep Hybrid-KNN_best"
    # Beispielscore: Mittelwert der Drift-Scores aus Window
    drift_score_value = drift_score 
    DRIFT_SCORE.labels(model=model_name).set(drift_score_value)
    with open(drift_prom, "w") as f:
        f.write(f"api_request_drift_alert {drift_flag}\n")
        f.write(f"api_request_drift_share {drift_share:.4f}\n")
        f.write(f'drift_score_sliding_window{{model="Deep Hybrid-KNN_best"}} {drift_score:.6f}\n')

    logger.info(f"""✅ Prometheus-Metriken gespeichert:
    api_request_drift_alert {drift_flag}
    api_request_drift_share {drift_share:.4f}
    drift_score_sliding_window{{model="Deep Hybrid-KNN_best"}} {drift_score:.6f}""")

except Exception as e:
    logger.error(f"❌ Fehler beim Schreiben der Prometheus-Metriken: {e}")