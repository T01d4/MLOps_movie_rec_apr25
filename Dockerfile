# Nutze ein schlankes, sauberes Python-Image
FROM python:3.12-slim

# Arbeitsverzeichnis im Container
WORKDIR /workspace

# Installiere Systemabhängigkeiten
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Erstelle Virtual Environment
ENV VIRTUAL_ENV=/workspace/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Kopiere requirements und installiere
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Aktiviere bash als Default-Shell
CMD ["bash"]
