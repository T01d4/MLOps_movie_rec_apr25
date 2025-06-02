#Dockerfile.api
FROM python:3.10-slim

# 1. Setze Arbeitsverzeichnis
WORKDIR /app

# 2. Installiere System-Dependencies (optional: falls du z. B. build-essential oder git brauchst)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Den gesamten App-Code kopieren (inkl. .env, falls benötigt)
COPY . .

# 5. Explizit Environment-File setzen (empfohlen: in .dockerignore nur sensible lokale Daten ausschließen)
ENV PYTHONUNBUFFERED=1

# 6. (Optional) Default-Port als ENV für Uvicorn/FastAPI (kann von außen überschrieben werden)
ENV PORT=8000

# 7. Start-Command (CMD als JSON, besser für Entrypoint-Parsing)
CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]