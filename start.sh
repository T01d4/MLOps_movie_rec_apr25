# #!/bin/bash
# set -e

# # Pfad zum Airflow logs-Verzeichnis (anpassen, falls anders)
# LOGS_DIR="./airflow/logs"

# echo "Adjusting permissions for folders ..."
# sudo chown -R $(id -u):$(id -g) .
# sudo chmod -R u+rwX,g+rwX,o+rX .
# # sudo chown -R 50000:50000 "."
# # sudo chmod -R 775 "."
# # sudo chown -R 50000:50000 ./data/raw
# # sudo chmod -R 775 ./data/raw
# # sudo chown -R 50000:50000 ./data/processed
# # sudo chmod -R 775 ./data/processed
# # sudo chown -R 50000:50000 ./models
# # sudo chmod -R 775 ./models
# # sudo chown -R 50000:50000 ./src
# # sudo chmod -R 775 ./src

# echo "Starting Docker Compose ..."
# docker compose up --build -d

# echo "Done."


## 2. Try

# #!/bin/bash
# set -e

# # Absolute Pfade anpassen auf dein Projektroot
# PROJECT_ROOT="/home/dev/projects/datascientest/MLOps_movie_rec_apr25"
# AIRFLOW_LOGS="$PROJECT_ROOT/airflow/logs"
# DATA_RAW="$PROJECT_ROOT/data/raw"
# DATA_PROCESSED="$PROJECT_ROOT/data/processed"
# MODELS="$PROJECT_ROOT/models"
# SRC="$PROJECT_ROOT/src"

# # Hier die UID/GID des Airflow-User im Container eintragen (z.B. 50000)
# AIRFLOW_UID=50000
# AIRFLOW_GID=50000

# echo "Fixing permissions recursively for Airflow project folders..."

# # 1. Falls du möchtest, kannst du auf deinen Host-User setzen (für lokale Entwicklung)
# # echo "Setting ownership to current user: $(id -u):$(id -g)"
# # sudo chown -R $(id -u):$(id -g) "$PROJECT_ROOT"
# # sudo chmod -R u+rwX,g+rwX,o+rX "$PROJECT_ROOT"

# # 2. Oder für produktiven Container-User (Airflow inside Docker)
# echo "Setting ownership to Airflow container user: $AIRFLOW_UID:$AIRFLOW_GID"
# sudo chown -R $AIRFLOW_UID:$AIRFLOW_GID "$AIRFLOW_LOGS" "$DATA_RAW" "$DATA_PROCESSED" "$MODELS" "$SRC"

# echo "Setting directory and file permissions..."
# sudo find "$PROJECT_ROOT" -type d -exec chmod 775 {} +
# sudo find "$PROJECT_ROOT" -type f -exec chmod 664 {} +

# echo "Permissions fixed."

# echo "Starting Docker Compose..."
# docker compose up --build -d

# echo "Done."



## 3. Try
#!/bin/bash
set -e

# Pfad zum Projekt-Root ermitteln (das Verzeichnis, wo diese start.sh liegt)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

AIRFLOW_UID=50000
AIRFLOW_GID=50000

echo "Fixing permissions recursively only for runtime folders..."

sudo chown -R $AIRFLOW_UID:$AIRFLOW_GID \
    "$PROJECT_ROOT/airflow/logs" \
    "$PROJECT_ROOT/data/raw" \
    "$PROJECT_ROOT/data/processed" \
    "$PROJECT_ROOT/models" \
    "$PROJECT_ROOT/src"

echo "Setting directory permissions (775) and file permissions (664) only in runtime dirs..."

for dir in "$PROJECT_ROOT/airflow/logs" "$PROJECT_ROOT/data/raw" "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/models" "$PROJECT_ROOT/src"
do
  sudo find "$dir" -type d -exec chmod 775 {} +
  sudo find "$dir" -type f -exec chmod 664 {} +
done

echo "Permissions fixed."

echo "Starting Docker Compose..."
docker compose up --build -d

echo "Done."
