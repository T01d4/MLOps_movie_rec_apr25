# src/monitoring/generate_embedding.py
import os
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger("airflow.task")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MONITOR_DIR = os.path.join(DATA_DIR, "monitoring")
#os.makedirs(MONITOR_DIR, exist_ok=True)
# === Load embedding ===
try:
    embedding_path = os.path.join(PROCESSED_DIR, "hybrid_deep_embedding.csv")
    embedding_df = pd.read_csv(embedding_path, index_col=0)
    logger.info(f"✅ Embedding loaded: {embedding_path}")
except Exception as e:
    logger.error(f"❌ Error loading embedding: {e}")
    raise

# === Shuffle and split ===
embedding_df = embedding_df.sample(frac=1.0, random_state=42)
n_train = int(len(embedding_df) * 0.8)
train_df = embedding_df.iloc[:n_train].copy()
test_df = embedding_df.iloc[n_train:].copy()

# Add target column for drift detection
train_df["target"] = 1
test_df["target"] = 1
reference_out = os.path.join(MONITOR_DIR, "reference_embedding.csv")
try:
    # Save only features (without "target")
    train_df.drop(columns=["target"]).to_csv(reference_out, index=False)
    logger.info(f"📌 Reference embedding saved: {reference_out}")
except Exception as e:
    logger.error(f"❌ Error saving reference embedding: {e}")
try:
    train_out = os.path.join(MONITOR_DIR, "train.csv")
    test_out = os.path.join(MONITOR_DIR, "test.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"✅ Snapshot saved (Train: {len(train_df)}, Test: {len(test_df)})")
except Exception as e:
    logger.error(f"❌ Error saving snapshot file: {e}")
    raise