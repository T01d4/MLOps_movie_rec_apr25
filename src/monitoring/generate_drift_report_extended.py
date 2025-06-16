# src/monitoring/generate_drift_report_extended.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# === Logging ===
logger = logging.getLogger("drift_report_extended")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# === Umgebungsvariablen ===
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
REPORT_DIR = os.getenv("REPORT_DIR", "/app/reports")
MONITOR_DIR = os.path.join(DATA_DIR, "monitoring")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# === Dateipfade ===
reference_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding_best.csv")
current_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
metrics_path = os.path.join(MONITOR_DIR, "metrics_from_mlflow.csv")
loss_path = os.path.join(REPORT_DIR, "training_loss_per_epoch.json")
output_path = os.path.join(REPORT_DIR, "drift_report_extended.html")

# === Ordner prüfen/erstellen ===


# === Daten laden ===
try:
    logger.info(f"✨ Lade Referenzdaten von {reference_path}")
    reference = pd.read_csv(reference_path, index_col=0).astype("float32")
    logger.info(f"✨ Lade aktuelle Daten von {current_path}")
    current = pd.read_csv(current_path, index_col=0).astype("float32")

    # Sicherstellen, dass Spalten übereinstimmen
    common_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[common_cols]
    current = current[common_cols]

except Exception as e:
    logger.error(f"❌ Fehler beim Laden der Daten: {e}")
    reference = pd.DataFrame()
    current = pd.DataFrame()

# === Metriken laden ===
try:
    logger.info(f"✨ Lade Metriken von {metrics_path}")
    metrics = pd.read_csv(metrics_path).sort_values("start_time", ascending=True)
except Exception as e:
    logger.warning(f"⚠️ Konnte Metriken nicht laden: {e}")
    metrics = pd.DataFrame()

# === Evidently Report erstellen ===
try:
    if reference.empty or current.empty:
        raise ValueError("Einer der DataFrames ist leer, kein Report möglich.")

    logger.info("📊 Generiere Evidently Report")
    column_mapping = ColumnMapping(numerical_features=common_cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    temp_output_path = output_path + ".tmp"
    report.save_html(temp_output_path)
    os.replace(temp_output_path, output_path)
    logger.info(f"✅ Drift Report gespeichert unter: {output_path}")
except Exception as e:
    logger.error(f"❌ Fehler beim Erzeugen des Reports: {e}")


def atomic_savefig(fig, final_path):
    base, ext = os.path.splitext(final_path)
    tmp_path = base + "_tmp" + ext
    fig.savefig(tmp_path)
    os.replace(tmp_path, final_path)
    
try:
    if "precision_10" in metrics.columns:
        fig, ax = plt.subplots()
        metrics.plot(x="start_time", y="precision_10", ax=ax, title="Precision@10 Verlauf", marker="o")
        ax.set_ylabel("Precision@10")
        atomic_savefig(fig, os.path.join(REPORT_DIR, "precision_10.png"))
        logger.info("📈 Precision@10 geplottet")

    if "drift_score" in metrics.columns:
        fig, ax = plt.subplots()
        metrics.plot(x="start_time", y="drift_score", ax=ax, title="Drift Score Verlauf", marker="o")
        ax.set_ylabel("Drift Score")
        atomic_savefig(fig, os.path.join(REPORT_DIR, "drift_score.png"))
        logger.info("📈 Drift Score geplottet")

    if "latency" in metrics.columns:
        fig, ax = plt.subplots()
        metrics.plot(x="start_time", y="latency", ax=ax, title="Inference Latency", marker="x")
        ax.set_ylabel("Latenz (s)")
        atomic_savefig(fig, os.path.join(REPORT_DIR, "latency.png"))
        logger.info("📈 Latenz geplottet")

    if os.path.exists(loss_path):
        with open(loss_path, "r") as f:
            loss_data = json.load(f)
        df_loss = pd.DataFrame(loss_data)
        fig, ax = plt.subplots()
        df_loss.plot(x="epoch", y="loss", ax=ax, title="Training Loss per Epoch", marker="*")
        ax.set_ylabel("Loss")
        atomic_savefig(fig, os.path.join(REPORT_DIR, "training_loss.png"))
        logger.info("📈 Training Loss geplottet")
except Exception as e:
    logger.warning(f"⚠️ Fehler bei der Visualisierung: {e}")
