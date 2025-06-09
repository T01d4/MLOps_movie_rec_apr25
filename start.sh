#!/bin/bash
set -e

# Pfad zum Airflow logs-Verzeichnis (anpassen, falls anders)
LOGS_DIR="./airflow/logs"

echo "Adjusting permissions for folders ..."
sudo chown -R 50000:50000 "./airflow/logs"
sudo chmod -R 775 "./airflow/logs"
sudo chown -R 50000:50000 ./data/raw
sudo chmod -R 775 ./data/raw


echo "Starting Docker Compose ..."
docker compose up --build -d

echo "Done."
