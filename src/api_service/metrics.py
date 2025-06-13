# api_service/metrics.py
from fastapi import APIRouter, Request, Response
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
import time
import os

# === Prometheus Metriken definieren ===
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Latency in seconds",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "error_count", "Total error responses",
    ["endpoint", "status_code"]
)

# === Router definieren ===
router = APIRouter()

# === /metrics Endpoint für Prometheus ===
@router.get("/metrics")
def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# === Drift-Metriken (.prom-Datei aus Airflow-Task oder TextfileExporter) ===
REPORT_DIR = os.environ.get("REPORT_DIR", "/app/reports")
DRIFT_METRICS_PATH = os.path.join(REPORT_DIR, "drift_metrics.prom")

@router.get("/metrics/drift")
def drift_metrics_endpoint():
    try:
        with open(DRIFT_METRICS_PATH, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        return Response(
            content="# drift_metrics.prom not found",
            media_type="text/plain",
            status_code=404
        )

# === Middleware zur Laufzeit-Überwachung ===
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    status_code = response.status_code

    REQUEST_COUNT.labels(request.method, endpoint, status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)
    if status_code >= 400:
        ERROR_COUNT.labels(endpoint, status_code).inc()

    return response