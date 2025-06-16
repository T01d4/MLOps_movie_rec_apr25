from fastapi import APIRouter, Request, Response
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
import time
import os

# === Prometheus Metriken definieren ===

# Gesamtanzahl aller Requests
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests",
    ["method", "endpoint", "status_code"]
)

# Latenzzeit in Sekunden
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Latency in seconds",
    ["endpoint"]
)

# Fehleranzahl nach Status-Code
ERROR_COUNT = Counter(
    "error_count", "Total error responses",
    ["endpoint", "status_code"]
)

# Gesundheitsstatus (1=OK, 0=Fehler)
HEALTH_STATUS = Gauge(
    "health_status", "1 = healthy, 0 = error"
)

# Optional: Simulierter Nutzerzähler (manuell inkrementieren)
UNIQUE_USERS = Gauge(
    "recommendation_unique_users_total", "Number of unique sessions/users"
)

# ML-Metriken wie Precision@10 aus dem Training (über Airflow nachtragbar)
PRECISION_AT_10 = Gauge(
    "model_precision_at_10", "Precision@10 score of best model",
    ["model"]
)

# === Router definieren ===
router = APIRouter()

# === /metrics Endpoint für Prometheus ===
@router.get("/metrics")
def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# === /metrics/drift lädt drift_metrics.prom aus Airflow ===
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

# === Healthcheck Endpoint (setzt auch Health Metric) ===
@router.get("/healthz")
def health_check():
    try:
        HEALTH_STATUS.set(1)  # OK
        return {"status": "ok"}
    except:
        HEALTH_STATUS.set(0)
        return {"status": "error"}

# === Middleware zur Laufzeit-Überwachung ===
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    status_code = response.status_code

    # Request zählen
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()

    # Latenz messen
    REQUEST_LATENCY.labels(endpoint).observe(latency)

    # Fehler zählen
    if status_code >= 400:
        ERROR_COUNT.labels(endpoint, str(status_code)).inc()

    return response