# monitoring/plot_precision_history.py

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

# === Pfade über Umgebungsvariablen ===
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
REPORT_DIR = os.getenv("REPORT_DIR", "/opt/airflow/reports")

METRICS_CSV_PATH = os.path.join(DATA_DIR, "monitoring", "metrics_from_mlflow.csv")
PRECISION_PLOT_PATH = os.path.join(REPORT_DIR, "precision_history.png")

# === Logging Setup ===
logger = logging.getLogger("airflow.task")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_precision_history(model_name="hybrid_deep_model", max_runs=20):
    try:
        client = MlflowClient()
        logger.info(f"📡 Verbindung zu MLflow-Client erfolgreich – Modell: {model_name}")

        runs = mlflow.search_runs(
            experiment_names=[model_name],
            max_results=max_runs
        )

        if runs.empty:
            logger.warning(f"⚠️ Keine Runs gefunden für Modell: {model_name}")
            return pd.DataFrame()

        runs = runs[["run_id", "start_time", "metrics.precision_10"]].dropna()
        runs["start_time"] = pd.to_datetime(runs["start_time"], unit="ms")

        sorted_runs = runs.sort_values("start_time", ascending=True)
        logger.info(f"✅ {len(sorted_runs)} MLflow-Runs mit Precision@10 geladen")
        return sorted_runs

    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der MLflow-Runs: {e}")
        return pd.DataFrame()

def plot_and_save_precision(df, output_path=PRECISION_PLOT_PATH):
    if df.empty:
        logger.warning(f"⚠️ Precision-Plot wird übersprungen – leeres DataFrame erhalten")
        return

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df["start_time"], df["metrics.precision_10"], marker="o")
        plt.title("Precision@10 Verlauf")
        plt.xlabel("Zeit")
        plt.ylabel("Precision@10")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"📈 Precision-Verlauf gespeichert unter: {output_path}")
    except Exception as e:
        logger.error(f"❌ Fehler beim Erstellen des Plots: {e}")

def save_metrics_csv(model_name="hybrid_deep_model", out_path=METRICS_CSV_PATH):
    try:
        client = MlflowClient()
        runs = mlflow.search_runs(experiment_names=[model_name], max_results=100)
        if runs.empty:
            logger.warning(f"⚠️ Keine Runs für {model_name} gefunden")
            return

        base_cols = ["run_id", "start_time"]
        wanted_metrics = [
            "metrics.precision_10",
            "metrics.inference_latency",
            "metrics.drift_score_sliding_window",
            "metrics.recommendation_requests_total"
        ]

        # Fallback-Spalten sicherstellen
        for m in wanted_metrics:
            if m not in runs.columns:
                runs[m] = np.nan

        metrics_df = runs[base_cols + wanted_metrics].copy()

        # Fehlende Werte ersetzen
        metrics_df["metrics.inference_latency"] = metrics_df["metrics.inference_latency"].fillna(0.0)
        metrics_df["metrics.drift_score_sliding_window"] = metrics_df["metrics.drift_score_sliding_window"].fillna(0.0)
        metrics_df["metrics.recommendation_requests_total"] = metrics_df["metrics.recommendation_requests_total"].fillna(0)

        metrics_df.rename(columns={
            "metrics.precision_10": "precision_10",
            "metrics.inference_latency": "latency",
            "metrics.drift_score_sliding_window": "drift_score",
            "metrics.recommendation_requests_total": "api_calls"
        }, inplace=True)

        metrics_df["start_time"] = pd.to_datetime(metrics_df["start_time"], unit="ms")
        metrics_df["version"] = metrics_df["run_id"].str[:8]
        export_cols = ["version", "start_time", "precision_10", "latency", "drift_score", "api_calls"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metrics_df.to_csv(out_path, index=False, columns=export_cols)

        logger.info(f"📄 CSV gespeichert: {out_path}")

    except Exception as e:
        logger.error(f"❌ Fehler beim Exportieren der Metriken: {e}")

if __name__ == "__main__":
    df = load_precision_history()
    plot_and_save_precision(df)
    save_metrics_csv()