from fastapi import APIRouter, Request, Response
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
import time
import os

# === Define Prometheus Metrics ===

# Total number of all requests
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests",
    ["method", "endpoint", "status_code"]
)

# Latency in seconds
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Latency in seconds",
    ["endpoint"]
)

# Total number of errors by status code
ERROR_COUNT = Counter(
    "error_count", "Total error responses",
    ["endpoint", "status_code"]
)

# Health status: 1 = OK, 0 = Error
HEALTH_STATUS = Gauge(
    "health_status", "1 = healthy, 0 = error"
)

# Optional: simulated unique user/session counter
UNIQUE_USERS = Gauge(
    "recommendation_unique_users_total", "Number of unique sessions/users"
)

# ML metrics like precision@10 (can be updated via Airflow)
PRECISION_AT_10 = Gauge(
    "model_precision_at_10", "Precision@10 score of best model",
    ["model"]
)

# === Initialize router ===
router = APIRouter()

# === /metrics endpoint for Prometheus scraping ===
@router.get("/metrics")
def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# === /metrics/drift serves drift_metrics.prom created by Airflow ===
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

# === /healthz endpoint for service readiness check ===
@router.get("/healthz")
def health_check():
    try:
        HEALTH_STATUS.set(1)  # OK
        return {"status": "ok"}
    except:
        HEALTH_STATUS.set(0)
        return {"status": "error"}

# === Monitoring middleware to track request metrics ===
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    status_code = response.status_code

    # Count requests
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()

     # Observe latency
    REQUEST_LATENCY.labels(endpoint).observe(latency)

    # Count errors
    if status_code >= 400:
        ERROR_COUNT.labels(endpoint, str(status_code)).inc()

    return response