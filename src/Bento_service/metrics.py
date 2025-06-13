#bento_service/metrics.py
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
import time

# Zählung der Requests
REQUEST_COUNT = Counter(
    "request_count",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

# Latenzmessung
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# Fehlerzählung
ERROR_COUNT = Counter(
    "error_count",
    "Count of failed requests",
    ["endpoint", "status_code"]
)

# Middleware zur Messung
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

# /metrics Endpoint
def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )