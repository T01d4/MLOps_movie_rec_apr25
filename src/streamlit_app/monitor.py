import json

with open("/app/data/monitoring/drift_metrics.json") as f:
    drift = json.load(f)