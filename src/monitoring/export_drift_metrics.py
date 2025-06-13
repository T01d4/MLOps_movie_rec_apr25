#monitoring/export_drift_metrics.py
import os
import json
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Pfade
REPORT_DIR = os.environ.get("REPORT_DIR", "/opt/airflow/reports")
DRIFT_JSON_PATH = os.path.join(REPORT_DIR, "drift_metrics.json")
PROM_FILE = os.path.join(REPORT_DIR, "drift_metrics.prom")
CONF_PATH = os.path.join(os.environ.get("DATA_DIR", "/opt/airflow/data"), "monitoring", "monitoring_conf.json")

# Lade Konfiguration
try:
    with open(CONF_PATH, "r", encoding="utf-8") as f:
        conf = json.load(f)
    drift_threshold = conf.get("drift_alert_threshold", 0.3)
    logging.info(f"📥 Konfiguration geladen (drift_alert_threshold={drift_threshold})")
except Exception as e:
    logging.warning(f"⚠️ Konfiguration konnte nicht geladen werden, nutze Default 0.3 – {e}")
    drift_threshold = 0.3

# Lade Evidently Report
try:
    with open(DRIFT_JSON_PATH, "r", encoding="utf-8") as f:
        drift_data = json.load(f)
    logging.info(f"✅ drift_metrics.json geladen von {DRIFT_JSON_PATH}")
except Exception as e:
    logging.error(f"❌ Fehler beim Laden von drift_metrics.json: {e}")
    drift_data = {"metrics": [{}]}

# Sicheres Extrahieren der Werte
def safe_get(metrics, idx, key, default=0.0):
    try:
        return metrics["metrics"][idx]["result"].get(key, default)
    except Exception as e:
        logging.warning(f"⚠️ Zugriff auf metrics[{idx}]['{key}'] fehlgeschlagen: {e}")
        return default

# Werte extrahieren
data_drift = safe_get(drift_data, 0, "dataset_drift")
target_psi = safe_get(drift_data, 1, "psi")
n_drifted = safe_get(drift_data, 0, "number_of_drifted_columns", 0)
share_drift = safe_get(drift_data, 0, "share_of_drifted_columns", 0.0)
drift_alert = int(data_drift > drift_threshold)

# Prometheus-Zeilen
lines = [
    f"data_drift_share {data_drift:.4f}",
    f"target_drift_psi {target_psi:.4f}",
    f"drifted_columns_total {int(n_drifted)}",
    f"drifted_columns_share {share_drift:.4f}",
    f"drift_alert {drift_alert}"
]

# Schreiben in Datei
try:
    with open(PROM_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logging.info(f"✅ Prometheus-Metriken gespeichert unter: {PROM_FILE}")
except Exception as e:
    logging.error(f"❌ Fehler beim Schreiben der Prometheus-Datei: {e}")
    raise